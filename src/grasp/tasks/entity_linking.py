import re
import unicodedata
from typing import Any

from pydantic import BaseModel

from grasp.configs import GraspConfig
from grasp.examples import Sample
from grasp.functions import find_manager, parse_iri_or_literal
from grasp.manager import KgManager, format_kgs
from grasp.model import Message
from grasp.sparql.types import Alternative, ObjType
from grasp.tasks.base import FeedbackTask, GraspTask
from grasp.tasks.entities import Entity, prepare_entity
from grasp.utils import (
    FunctionCallException,
    format_enumerate,
    format_list,
    format_notes,
    format_section,
)


class TextAnnotation(Entity):
    start_index: int
    end_index: int


class Text(BaseModel):
    data: str
    annotate_from: int | None = None
    annotate_up_to: int | None = None
    special_instructions: str | None = None

    @property
    def length(self) -> int:
        return len(self.data)

    @property
    def start(self) -> int:
        return self.annotate_from if self.annotate_from is not None else 0

    @property
    def end(self) -> int:
        return (
            self.annotate_up_to if self.annotate_up_to is not None else len(self.data)
        )

    def trim(self, context: int | None = None) -> tuple["Text", int]:
        # Trims the Text to the start/end values if context is 0, trims the Text to
        # start/end plus context otherwise. If context is None, does not trim the Text.
        if context and context < 0:
            raise ValueError(f"context '{context}' must be non negative.")
        if self.start < 0 or self.start >= self.length:
            raise ValueError(
                f"annotate_from '{self.start}' must be greater than or equal to zero "
                f"and less than length '{self.length}'."
            )
        if self.end <= self.start or self.end > self.length:
            raise ValueError(
                f"annotate_up_to '{self.end}' must be greater than annotate_from "
                f"'{self.start}' and less than or equal to length '{self.length}'."
            )

        # without context the text is not trimmed
        if context is None:
            return self, 0

        # 4 variables: start/end of new context and start/end of new annotation window
        new_start, new_end = self.start, self.end
        window_start = 0

        if self.start > 0:
            new_start = max(0, self.start - context)
            window_start = self.start - new_start

        if self.end < self.length:
            new_end = min(self.length, self.end + context)

        window_end = self.end - new_start

        trimmed = Text(
            data=self.data[new_start:new_end],
            annotate_from=window_start,
            annotate_up_to=window_end,
        )
        return trimmed, new_start


class EntityLinkingSample(Sample):
    text: Text
    annotations: list[TextAnnotation]

    def input(self) -> Any:
        return self.text.model_dump()

    def queries(self) -> list[str]:
        annots = AnnotationState(self.text)
        return [annots.format()]


class AnnotationState:
    def __init__(
        self,
        text: Text,
        context: int | None = None,
        method: str = "matching",
    ) -> None:
        self.text, self.offset = text.trim(context)
        self.annotation_window: slice = slice(self.text.start, self.text.end)
        self.annotations: dict[tuple[int, int], Annotation] = {}
        self.word_indices: dict[int, tuple[int, int]] = {}

    def annotate(
        self,
        start_index: int,
        end_index: int,
        annotation: Entity | None,
    ) -> Entity | None:
        aws = self.annotation_window.stop - self.annotation_window.start
        if start_index < 0 or start_index >= aws:
            raise ValueError(f"Start_index {start_index} out of bounds")
        if end_index <= start_index or end_index > aws:
            raise ValueError(f"End_index {end_index} out of bounds")
        start_index += self.annotation_window.start
        end_index += self.annotation_window.start
        current = self.annotations.pop((start_index, end_index), None)
        if annotation is not None:
            self.annotations[(start_index, end_index)] = annotation
        return current

    def get(self, start_index: int, end_index: int) -> Entity | None:
        return self.annotations.get((start_index, end_index), None)

    def to_dict(self) -> dict:
        return {
            "formatted": self.format(),
            "predictions": [
                {
                    "entity_reference": a.entity,
                    "start_char": s + self.offset,
                    "end_char": e + self.offset,
                    "identifier": a.identifier,
                    "label": a.label,
                }
                for (s, e), a in self.annotations.items()
            ],
        }



    def format(
        self,
        only_current_window: bool = False,
        list_entities: bool = False,
        add_word_indices: bool = False,
    ) -> str:
        """
        Returns a string with the current annotation state of the text.
        Annotations are visualized in the following format: '[annotated words](q123)',
        '[[Nested [annotations](q123)](q456) are supported](q789)'.
        If only_current_window is true, only the text of the current annotation window
        is shown. If add_word_indices is true, the function 'create_word_indices' is
        used to generate indices for words which are then shown in the folling format:
        'word(1) and(2) composed(3)-word(4) and [annotated(5) words(6)](q123).'
        """
        result = self.text.data
        if add_word_indices:
            if self.word_indices == {}:
                self.create_word_indices()
            sorted_indices = sorted(
                self.word_indices.items(),
                key=lambda item: item[1][1]
            )
        else:
            sorted_indices = {}

        # item[0] is (start, end), we sort by end first, then by negative start
        sorted_annotations = sorted(
            self.annotations.items(),
            key=lambda item: (item[0][1], -item[0][0])
        )



        # go through annotations from highest end index first
        nested_list = []
        while sorted_annotations or sorted_indices:
            currently_annotating_index = False
            if (not sorted_annotations or
                sorted_indices and sorted_annotations
                and sorted_indices[-1][1][1] + self.annotation_window.start
                > sorted_annotations[-1][0][1]
            ):
                currently_annotating_index = True
                ind = sorted_indices.pop() 
                start_idx = end_idx = ind[1][1] + self.annotation_window.start
            else:
                ann = sorted_annotations.pop() 
                start_idx = ann[0][0]
                end_idx = ann[0][1]

            start_offset = 0
            end_offset = 0
            for i in range(len(nested_list) -1, -1, -1):
                # start of other annotation after end of current one -> unimportant
                if nested_list[i] >= end_idx:
                    nested_list.pop(i)
                # start of other annotation before current end but not current start
                elif nested_list[i] < end_idx and nested_list[i] >= start_idx:
                    end_offset += 1
                # we don't need to see the rest of the list
                elif nested_list[i] < start_idx:
                    start_offset += 1
                    end_offset += 1
            # prepend current start to the nested list
            nested_list = [start_idx] + nested_list

            if (
                only_current_window
                and (start_idx < self.annotation_window.start
                or end_idx > self.annotation_window.stop)
            ):
                continue

            prefix = result[:start_idx + start_offset]
            words = result[start_idx + start_offset:end_idx + end_offset]
            suffix = result[end_idx + end_offset:]
            if currently_annotating_index:
                result = (prefix + "⟦" + str(ind[0]) + "⟧" + suffix)
            else:
                result = (prefix + "[" + words + "](" + ann[1].entity + ")" + suffix)


        # trim to only show the current window
        if only_current_window:
            added_length = len(result) - self.text.length
            result = result[
                self.annotation_window.start : self.annotation_window.stop
                + added_length
            ]

        if list_entities:
            entities: dict[str, Alternative] = {}
            for annot in self.annotations.values():
                if annot.identifier in entities:
                    continue

                entities[annot.identifier] = annot.to_alternative()

            if entities:
                annotations = format_list(
                    alt.get_selection_string() for _, alt in sorted(entities.items())
                )
                result += f"\n\nAnnotated entities:\n{annotations}"

        return result


    def create_word_indices(self) -> None:
        """
        A heuristic to create indices for words in order to enable the index 
        annotation function, changes self.word_indices.
        """
        #if self.method != "indices":
        #    raise FunctionCallException(
        #        "method is not 'indices', function create_indices should not be called"
        #    )

        currently_at_word = False
        currently_at_number = False
        self.word_indices = {}
        idx = 0
        current_excerpt = self.text.data[self.annotation_window]

        number_characters = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
        number_punctuation = {".", ","}
        punctuation_characters = {
            " ", ".", ",", "-", ":", ";", "[", "]", "(", ")", "{", "}", "\n", "'", "\"", "’", "‘", "?", "="
        }

        for i, c in enumerate(current_excerpt):
            if not (c in punctuation_characters or currently_at_word or currently_at_number): 
                s = i
                if c in number_characters:
                    currently_at_number = True
                else:
                    currently_at_word = True

            # ignore '.' or ',' in the middle of number
            elif (
                c in number_punctuation and currently_at_number 
                and len(current_excerpt) > i and current_excerpt[i + 1] in number_characters
            ):
                continue

            # end of word
            elif c in punctuation_characters and currently_at_word:
                currently_at_word = False
                e = i
                self.word_indices[idx] = (s, e)
                idx += 1

            # end of number
            elif c not in number_characters and currently_at_number:
                currently_at_number = False
                e = i
                self.word_indices[idx] = (s, e)
                idx += 1


    def delete_annotations_in_current_window(self):
        delete_list = []
        for ann in self.annotations:
            if (ann[0] >= self.annotation_window.start 
                or ann[1] < self.annotation_window.stop):
                delete_list += [ann]
        for ann in delete_list:
            self.annotations.pop(ann)


def rules() -> list[str]:
    return [
        (
            "If you cannot find any suitable entity mention in the text excerpt, "
            "leave the excerpt unannotated and just finalize."
        ),
        (
            "If an entity is cut off at the end of the text or beginning of the text, "
            "don't annotate it. Use the context to see if it is cut off or not."
        ),
        (
            "If there are multiple suitable entities for a number of words, choose "
            "the one that fits best in the context of the text and is more general."
        ),
        (
            "Annotate every occurence of an entity you find in the excerpt even if it "
            "is mentioned multiple times. If it occurs again, annotate it again."
        ),
        (
            "If you recognize an entity but cannot find it in the knowledge graph, "
            "annotate it as null."
        ),
        (
            "Do not link coreferences that do not contain at least part of the name "
            "but **do** link entity mentions that contain at least a part of a name."
        ),
        "If the user specifies additional instructions follow those instructions.",
    ]


def system_information() -> str:
    return """\
You are an entity linking system that does entity recognition and entity disambiguation according to the provided rules and instructions.
Your task is to annotate words in a given text excerpt with entities from the available knowledge graphs.

You need to **exactly follow these step-by-step instructions** to annotate the text:
1. Find entity mentions in the text excerpt by going through it from the begining word by word and determine if it constitutes an entity according to the rules.
2. Determine what the text might be about and think about how the entity mentions might be represented with entities in the knowledge graph(s). 
3. Use the provided functions to search and explore the knowledge graph(s) to find the entities.
4. Use the annotate function to annotate every entity mention.
5. When you are certain that the annotation of the current excerpt is correct and complete use the finalize function."""


def functions(managers: list[KgManager], config: GraspConfig) -> list[dict]:
    kgs = [manager.kg for manager in managers]
    task_kwargs = config.task_kwargs.get("entity-linking", {})
    method = task_kwargs.get("method", "matching")
    if method == "matching":
        fns = [
            {
                "name": "annotate",
                "description": """\
Annotate a word or a sequence of words with an entity from the specified knowledge \
graph by writing the exact words to be annotated as 'words_to_be_annotated'.
Specify the words further by the occurrence_index, if the words only occur once, set it to 0,
if you only want to annotate the second occurence, just set it to 1.
Careful, sometimes a word you want to annotate can be a substring of another word earlier in the text excerpt, \
so always keep that in mind and adjust the occurrence_index accordingly.
This function overwrites any previous annotation of the words.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph to use for the annotation.",
                    },
                    "words_to_be_annotated": {
                        "type": "string",
                        "description": "The exact words to be annotated written exactly like in the original text.",
                    },
                    "occurrence_index": {
                        "type": "integer",
                        "description": "Index of the occurrence of the words in the text.",
                    },
                    "entity": {
                        "type": ["string", "null"],
                        "description": "The IRI of the entity to annotate the words with, or null if unknown.",
                    },
                },
                "required": [
                    "kg",
                    "words_to_be_annotated",
                    "occurrence_index",
                    "entity",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "delete_annotation",
            "description": """\
Delete the annotation of a word or a sequence of words \
by writing the exact words whose annotation should be deleted as 'words_to_be_annotated'.
Specify the words further by the occurrence_index, if the words only occur once, set it to 0,
if you only want to delete the annotation of the second occurence, just set it to 1.
Careful, sometimes an annotation you want to delete can be a substring of another word earlier in the text excerpt, \
so always keep that in mind and adjust the occurrence_index accordingly.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "words_to_be_annotated": {
                            "type": "string",
                            "description": "The exact words whose annotation should be deleted, written exactly like in the original text.",
                        },
                        "occurrence_index": {
                            "type": "integer",
                            "description": "Index of the occurrence of the words in the text.",
                        },
                    },
                    "required": ["words_to_be_annotated", "occurrence_index"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]
    elif method == "prefix":
        fns = [
            {
                "name": "annotate",
                "description": """\
Annotate a word or a sequence of words with an entity from the specified knowledge \
graph by writing the exact words to be annotated as 'words_to_be_annotated'.
If the annotation fails you can input EITHER a couple of words before the words to be \
annotated as prefix OR you can input a couple words after as suffix.\
Do not use Newlines as suffix or prefix, use the next word in those cases.\
This function overwrites any previous annotation of the words.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph to use for the annotation",
                        },
                        "optional_short_prefix": {
                            "type": "string",
                            "description": "OPTIONAL: a word or two before the words to be annotated",
                        },
                        "exact_words_to_be_annotated": {
                            "type": "string",
                            "description": "The exact words to be annotated written exactly like in the original text",
                        },
                        "optional_short_suffix": {
                            "type": "string",
                            "description": "OPTIONAL: a word or two after the words to be annotated",
                        },
                        "entity": {
                            "type": ["string", "null"],
                            "description": "The IRI of the entity to annotate the words with, or null if unknown.",
                        },
                    },
                    "required": ["kg", "optional_short_prefix", "exact_words_to_be_annotated", "optional_short_suffix", "entity"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "delete_annotation",
                "description": """\
Delete the annotation of a word or a sequence of words by writing the \
exact words whose annotation should be deleted as 'words_to_be_annotated'.
If the annotation fails you can input EITHER a couple of words before the words to be \
annotated as prefix OR you can input a couple words after as suffix.\
Do not use Newlines as suffix or prefix, use the next word in those cases.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "optional_short_prefix": {
                            "type": "string",
                            "description": "OPTIONAL: a word or two before the words to be annotated",
                        },
                        "exact_words_to_be_annotated": {
                            "type": "string",
                            "description": "The exact words to be annotated written exactly like in the original text",
                        },
                        "optional_short_suffix": {
                            "type": "string",
                            "description": "OPTIONAL: a word or two after the words to be annotated",
                        },
                    },
                    "required": ["optional_short_prefix", "exact_words_to_be_annotated", "optional_short_suffix"],
                    "additionalProperties": False,
                },
                "strict": True,
            },]
    elif method == "indices":
        fns = [
            {
                "name": "annotate",
                "description": """\
Annotate a word or a sequence of words with an entity from the specified knowledge \
graph by inputing the index that you see in '⟦⟧' brackets behind the words to be \
annotated as 'start_index' and 'end_index' (they are the same if its only one word).
This function overwrites any previous annotation of the words.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph to use for the annotation",
                        },
                        "start_index": {
                            "type": "integer",
                            "description": "The start index of the words to be annotated",
                        },
                        "end_index": {
                            "type": "integer",
                            "description": "The end index of the words to be annotated (inclusive)",
                        },
                        "entity": {
                            "type": ["string", "null"],
                            "description": "The IRI of the entity to annotate the words with, or null if unknown.",
                        },
                    },
                    "required": ["kg", "start_index", "end_index", "entity"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "name": "delete_annotation",
                "description": """\
    Delete the annotation of a word or a sequence of words by inputing \
    the index that you see in '⟦⟧'brackets behind the words to be annotated as\
    'start_index' and 'end_index' (they are the same if its only one word).""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_index": {
                            "type": "integer",
                            "description": "The start index of the words to be annotated",
                        },
                        "end_index": {
                            "type": "integer",
                            "description": "The end index of the words to be annotated (inclusive)",
                        },
                    },
                    "required": ["start_index", "end_index"],
                    "additionalProperties": False,
                },
                "strict": True,
            },]
    elif method == "markdown":
        fns = [
            {
                "name": "annotate",
                "description": """\
Annotate a word or a sequence of words with an entity from the specified knowledge \
graph by annotating the words in the following format: ⟦words to be annotated⟧(entity id). \
So write a '⟦' before the words to be annotated, then '⟧' and immediately after '(' and the \
entity ID and then ')'. Like an inline markdown link format but with other brackets.  \
You need to annotate the full window in one go. Every call overwrites the old annotations. \
The parsing just looks for the combination of brackets and the annotation and then annotates \
according to the positions those are in, so the actual characters dont matter but the absolute \
position does. Keep that in mind if you're having trouble annotating, \
the first character could be a space that you don't see. \
This function overwrites any previous annotation of the words.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kg": {
                            "type": "string",
                            "enum": kgs,
                            "description": "The knowledge graph to use for the annotation",
                        },
                        "text_to_be_annotated": {
                            "type": "string",
                            "description": "The whole text to be annotated (your current excerpt) with the words to be annotated \
                            in '⟦' and '⟧' brackets followed by the entity id in '(' and ')' brackets \
                            e.g.: ⟦words to be annotated⟧(entity id)"
                        },
                    },
                    "required": ["kg", "text_to_be_annotated"],
                    "additionalProperties": False,
                },
                "strict": True,
            },]
    else:
        raise ValueError(
            f"annotation method '{method}' needs to be one of: matching, prefix, indices, markdown"
        )
    
    fns.extend([
        {
            "name": "show_current_annotations",
            "description": "Show the current annotation state of the excerpt of the text to annotate.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "finalize",
            "description": "Finalize your annotations in the given excerpt and stop the annotation process.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ])
    return fns


def handle_annotation(
    manager: KgManager,
    entity: str | None,
    start_idx: int,
    end_idx: int,
    state: set[str],
    known: set[str],
    know_before_annotate: bool,
    show_state_after_annotate: bool,
    add_word_indices: bool = False,
    ) -> str:
    try:
        if entity is None or entity == "<NIL>":
            annotation = Entity(identifier="<NIL>", entity="<NIL>")
        else:
            annotation = prepare_entity(manager, entity)
            if know_before_annotate and annotation.identifier not in known:
                raise FunctionCallException(
                    f"The entity {entity} cannot be used for annotation "
                    "without being known from previous function call results. "
                    "This does not mean it is invalid, but you should verify "
                    "that it indeed exists (e.g., by listing example triples) "
                    "in the knowledge graphs first."
                )

        current = state.annotate(start_idx, end_idx, annotation)

    except ValueError as e:
        raise FunctionCallException(str(e)) from e

    sequence = state.text.data[state.annotation_window]
    if current is None:
        result = (
            f"Annotated text sequence [{start_idx}: {end_idx}] "
            f"'{sequence[start_idx:end_idx]}' with entity '{entity}'."
        )
    else:
        result = (
            f"Updated annotation of text sequence [{start_idx}, {end_idx}] "
            f"'{sequence[start_idx:end_idx]}' from '{current.entity}' to '{entity}'."
        )
    if show_state_after_annotate:
        result += (
            "\n\nThe current annotation state of the text excerpt is the following:\n"
            f"{state.format(True, False, add_word_indices)}"
        )
    return result


def handle_deleting_annotation(
    state: AnnotationState,
    start_idx: int,
    end_idx: int,
    show_state_after_annotate: bool,
    add_word_indices: bool=False,
) -> str:
    try:
        current = state.annotate(start_idx, end_idx, None)
    except ValueError as e:
        raise FunctionCallException(str(e)) from e

    sequence = state.text.data[state.annotation_window]

    if current is None:
        raise FunctionCallException(
            f"Text sequence [{start_idx}, {end_idx}] '{sequence[start_idx:end_idx]}' "
            "is not annotated so there is no annotation to delete."
        )

    result = (
        f"Deleted annotation '{current.entity}' from text sequence "
        f"[{start_idx}, {end_idx}] '{sequence[start_idx:end_idx]}'."
    )
    if show_state_after_annotate:
        result += (
            "\n\nThe current annotation state of the text excerpt is the following:\n"
            f"{state.format(True, False, add_word_indices)}"
        )
    return result


def find_matches_for_matching_method(
    words_to_be_annotated: str,
    occurrence_index: int,
    sequence: str,
) -> tuple[int, int]:
    # normalizing, because some llms are heavily biased towards specific characters like
    # the ascii apostrophe although they are technically able to output the correct one.
    def normalize(string: str) -> str:
        return string.replace("‘", "'").replace("’", "'")

    words_to_be_annotated = normalize(words_to_be_annotated)
    sequence = normalize(sequence)

    word_matches = [
        m.span() for m in re.finditer(re.escape(words_to_be_annotated), sequence)
    ]

    if not word_matches:
        raise ValueError(
            f"No match found for the given words to be annotated "
            f"'{words_to_be_annotated}' in the current annotation window."
            "(Did you use the correct characters when specifying the words?)"
        )

    if occurrence_index < 0:
        raise ValueError(f"occurrence_index '{occurrence_index}' must be non negative.")

    if occurrence_index >= len(word_matches):
        raise ValueError(
            f"occurrence_index '{occurrence_index}' must be less than "
            f"number of matches: {len(word_matches)}."
        )

    return word_matches[occurrence_index]


def annotate_matching(
    managers: list[KgManager],
    kg: str,
    words_to_be_annotated: str,
    occurrence_index: int,
    entity: str | None,
    state: AnnotationState,
    known: set[str],
    know_before_annotate: bool = False,
    show_state_after_annotate: bool = True,
) -> str:
    # A function for the llm to call to annotate the words_to_be_annotated in the text
    # with the entity and knowledge graph. The occurrence_index helps to distinguish
    # between different occurrences of the words in the text excerpt.
    manager, _ = find_manager(managers, kg)
    sequence = state.text.data[state.annotation_window]

    start_idx, end_idx = find_matches_for_matching_method(
        words_to_be_annotated, occurrence_index, sequence
    )

    return handle_annotation(
        manager,
        entity,
        start_idx,
        end_idx,
        state,
        known,
        know_before_annotate,
        show_state_after_annotate,
    )

def delete_annotation_matching(
    words_to_be_annotated: str,
    occurrence_index: int,
    state: AnnotationState,
    show_state_after_annotate: bool = True,
) -> str:
    # A function for the llm to call to delete the annotation of the
    # words_to_be_annotated in the text. The occurrence_index helps to
    # distinguish between different occurrences of the words in the text.
    sequence = state.text.data[state.annotation_window]

    start_idx, end_idx = find_matches_for_matching_method(
        words_to_be_annotated, occurrence_index, sequence
    )

    return handle_deleting_annotation(
        state,
        start_idx,
        end_idx,
        show_state_after_annotate,
    )


def find_matches_for_prefix_method(
    prefix: str,
    words_to_be_annotated: str,
    suffix: str,
    sequence: str,
) -> tuple[int, int]:

    def normalize(string: str) -> str:
        return string.replace("‘", "'").replace("’", "'")

    words_to_be_annotated = normalize(words_to_be_annotated)
    sequence = normalize(sequence)
    prefix = normalize(prefix)
    suffix = normalize(suffix)
    sequence_length = len(sequence)
    prefix_length = len(prefix)
    suffix_length = len(suffix)

    word_matches = [
        m.span() for m in re.finditer(re.escape(words_to_be_annotated), sequence)
    ]

    if not word_matches:
        raise ValueError(
            f"No match found for the given words to be annotated "
            f"'{words_to_be_annotated}' in the current annotation window."
        )

    pm_cntr = 0
    pm_list = []
    # ignore these characters in between words and prefix/suffix since the 
    # llm sometimes doesn't consider them to be words and ommits them
    ignored_characters = {
        " ", "\n", "\r", ".", ",", "\"", "'", ";", ":", "“", "”", "„", "´", "’"
    }
    for pm in word_matches:
        start_idx, end_idx = pm
        m = 0
        n = 0
        brk = False
        while start_idx >= prefix_length + m:
            if prefix == sequence[start_idx - prefix_length - m: start_idx - m]:
                while sequence_length >= end_idx + suffix_length + n:
                    if suffix == sequence[end_idx + n: end_idx + suffix_length + n]:
                        pm_cntr += 1
                        start, end = pm
                        pm_list.append(pm)
                        brk = True
                        break
                    if sequence[end_idx + n] not in ignored_characters:
                        brk = True
                        break
                    n += 1
            m += 1
            if sequence[start_idx - m] not in ignored_characters or brk:
                break


    if pm_cntr < 1:
        raise ValueError(
            f"Match found for words to be annotated '{words_to_be_annotated}' but "
            f"no match found for the given prefix '{prefix}' or suffix '{suffix}'."
            "Maybe try leaving out either the prefix or suffix."
        )

    if pm_cntr > 1:
        raise ValueError(
            f"{pm_cntr} possible matches found for the given prefix '{prefix}' "
            f"and words '{words_to_be_annotated}' and suffix '{suffix}'."
            "Try adding a word for context either in the prefix or suffix."
        )

    return start, end


def annotate_prefix(
    managers: list[KgManager],
    kg: str,
    prefix: str,
    words_to_be_annotated: str,
    suffix: str,
    entity: str,
    state: AnnotationState,
    known: set[str],
    know_before_annotate: bool = False,
    show_state_after_annotate: bool = True,
) -> str:
    """
    A function for the llm to call to annotate the text with the exact string to be 
    annotated and optionally prefix and/or suffix to distinguish between different
    occurrences of the words in the original text.
    """
    manager, _ = find_manager(managers, kg)
    sequence = state.text.data[state.annotation_window]

    start_idx, end_idx = find_matches_for_prefix_method(
        prefix, words_to_be_annotated, suffix, sequence
    )

    return handle_annotation(
        manager,
        entity,
        start_idx,
        end_idx,
        state,
        known,
        know_before_annotate,
        show_state_after_annotate,
    )


def delete_annotation_prefix(
    prefix: str,
    words_to_be_annotated: str,
    suffix: str,
    state: AnnotationState,
    show_state_after_annotate: bool = True,
) -> str:
    """
    A function for the llm to call to annotate the text with the exact string to be 
    annotated and optionally prefix and/or suffix to distinguish between different
    occurrences of the words in the original text.
    """
    manager, _ = find_manager(managers, kg)
    sequence = state.text.data[state.annotation_window]

    start_idx, end_idx = find_matches_for_prefix_method(
        prefix, words_to_be_annotated, suffix, sequence
    )

    return handle_deleting_annotation(
        state,
        start_idx,
        end_idx,
        show_state_after_annotate,
    )


def annotate_indices(
    managers: list[KgManager],
    kg: str,
    start: int,
    end: int,
    entity: str | None,
    state: AnnotationState,
    known: set[str],
    know_before_annotate: bool = False,
    show_state_after_annotate: bool = True,
) -> str:
    # Annotation function that uses state.word_indices to translate word indices to 
    # character indices in the text excerpt and annotates the entity from the kg
    manager, _ = find_manager(managers, kg)

    if start not in state.word_indices:
        raise ValueError(f"start_index '{start}' is not a valid word index")
    if end not in state.word_indices:
        raise ValueError(f"end_index '{end}' is not a valid word index")
    start_idx = state.word_indices[start][0]
    end_idx = state.word_indices[end][1]

    return handle_annotation(
        manager,
        entity,
        start_idx,
        end_idx,
        state,
        known,
        know_before_annotate,
        show_state_after_annotate,
        add_word_indices=True,
    )


def delete_annotation_indices(
    start: int,
    end: int,
    state: AnnotationState,
) -> str:
    # Annotation function that uses state.word_indices to translate word indices to 
    # character indices in the text excerpt and deletes the annotation
    manager, _ = find_manager(managers, kg)

    if start not in state.word_indices:
        raise ValueError(f"Start_index {start} not a valid word index")
    if end not in state.word_indices:
        raise ValueError(f"End_index {end} not a valid word index")
    start_idx = state.word_indices[start][0]
    end_idx = state.word_indices[end][1]

    return handle_deleting_annotation(
        state,
        start_idx,
        end_idx,
        show_state_after_annotate,
        add_word_indices=True,
    )


def annotate_markdown(
    managers: list[KgManager],
    kg: str,
    text_to_be_annotated: str,
    state: AnnotationState,
    known: set[str],
    know_before_annotate: bool = False,
    show_state_after_annotate: bool = True,
) -> str:
    # Annotation function that works by inputing the whole text excerpt with annotations
    # in the format: 'original text ⟦words to be annotated⟧⟦q123⟧ rest of original text'
    # to annotate the text. Deletes all previous annotations in the current text excerpt.
    manager, _ = find_manager(managers, kg)

    sequence = state.text.data[state.annotation_window]
    # replace '⟦' and '⟧' in the input text with '[' and ']' to get a canonical format
    # TODO: put this in state.format...
    # text_to_be_annotated = text_to_be_annotated.replace("⟦", "[").replace("⟧", "]")
    annotation_length = len(text_to_be_annotated)

    potential_start = None
    found_middle = False
    self_correction_range = 10
    start_idx = 0
    end_idx = 0
    correction_amount = 1
    result = ""
    state.delete_annotations_in_current_window()
    for i in range(annotation_length):
        # if we find a ⟦ bracket, we remember its index
        if  not found_middle and text_to_be_annotated[i] == "⟦":
            potential_start = i + 1

        # if we have already found a starting bracket and now the combination ⟧⟦,
        # we know it could be an annotation
        elif potential_start is not None and (text_to_be_annotated[i-1: i+1] == "⟧("):
            found_middle = True
            start_idx = potential_start
            end_idx = i-1
            words = text_to_be_annotated[start_idx: end_idx]

        # if we find a round closing bracket after already finding the other prerequisites
        elif text_to_be_annotated[i] == ")" and found_middle:
            entity = text_to_be_annotated[end_idx+2: i]
            found_middle = False
            potential_start = None

            start_idx -= correction_amount
            end_idx -= correction_amount

            # try to recover at annotations if the input is only shorter/longer 
            # than the original text by self_correction_range number of characters
            if sequence[start_idx: end_idx] != words:
                for c in range(self_correction_range):
                    if sequence[start_idx - c: end_idx - c] == words:
                        correction_amount += c
                        start_idx -= c
                        end_idx -= c
                        result += f"\n- corrected {c} characters to the left"
                        break
                    if sequence[start_idx + c: end_idx + c] == words:
                        correction_amount -= c
                        start_idx += c
                        end_idx += c
                        result += f"\n- corrected {c} characters to the right"
                        break

            try:
                result +=  "\n- " + handle_annotation(
                    manager,
                    entity,
                    start_idx,
                    end_idx,
                    state,
                    known,
                    know_before_annotate,
                    show_state_after_annotate=False,
                )
        
            except Exception as e:
                result += f"\n- Error occured while annotating: {str(e)}"

            correction_amount = (i - end_idx + 2)

    if show_state_after_annotate:
        result += (
            "\n\nThe current annotation state of the text excerpt is the following:\n" 
            f"{state.format(only_current_window=True, list_entities=False)}"
        )
    return result


def input_instructions(
    state: AnnotationState,
    special_instructions: str | None = None,
    add_word_indices: bool = False,
) -> str:
    user_input = (
        "This is the full text only for context to better understand entities "
        "that are not clear from the excerpt alone.\n\n"
        "=== START FULL TEXT FOR CONTEXT ===\n"
        f"{state.format(only_current_window=False)}\n"
        "=== END FULL TEXT FOR CONTEXT ===\n\n"
        "The following is the excerpt of the text that you need to annotate:\n\n"
        "=== START TEXT EXCERPT TO ANNOTATE ===\n"
        f"{state.format(only_current_window=True, add_word_indices=add_word_indices)}\n"
        "=== END TEXT EXCERPT TO ANNOTATE ===\n"
    )
    if special_instructions:
        user_input = (
            "These are additional instructions that you need to follow.\n\n"
            "=== START ADDITIONAL INSTRUCTIONS ===\n"
            f"{special_instructions}\n"
            "=== END ADDITIONAL INSTRUCTIONS ===\n\n"
        ) + user_input
    return user_input


def input_and_state(
    input: Any,
    config: GraspConfig,
) -> tuple[str, AnnotationState]:
    try:
        text = Text(**input)
    except Exception as e:
        raise ValueError(
            "Entity Linking task input must be a dict with a 'data' and optional "
            "'annotate_from', 'annotate_up_to', and 'special_instructions' fields."
        ) from e

    el_kwargs = config.task_kwargs.get("entity-linking", {})
    annots = AnnotationState(text, context=el_kwargs.get("context"))
    method = el_kwargs.get("method", "")
    add_word_indices = method == "indices"
    instructions = input_instructions(
        annots,
        text.special_instructions,
        add_word_indices
    )

    return instructions, annots


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: AnnotationState | None = None,
    example_indices: dict | None = None,
) -> str:
    assert isinstance(state, AnnotationState), (
        "Annotations must be provided as state for entity linking task"
    )
    assert not example_indices, (
        "Example indices are not supported for entity linking task"
    )

    el_kwargs = config.task_kwargs.get("entity-linking", {})
    know_before_annotate = el_kwargs.get("know_before_annotate", True)
    show_state_after_annotate = el_kwargs.get("show_state_after_annotate", True)
    method = el_kwargs.get("method", "matching")

    if fn_name == "annotate":
        if method == "matching":
            return annotate_matching(
                managers,
                fn_args["kg"],
                fn_args["words_to_be_annotated"],
                fn_args["occurrence_index"],
                fn_args["entity"],
                state,
                known,
                know_before_annotate,
                show_state_after_annotate,
            )
        if method == "prefix":
            return annotate_prefix(
                managers,
                fn_args["kg"],
                fn_args["optional_short_prefix"],
                fn_args["exact_words_to_be_annotated"],
                fn_args["optional_short_suffix"],
                fn_args["entity"],
                state,
                known,
                know_before_annotate,
                show_state_after_annotate,
            )
        if method == "indices":
            return annotate_indices(
                managers,
                fn_args["kg"],
                fn_args["start_index"],
                fn_args["end_index"],
                fn_args["entity"],
                state,
                known,
                know_before_annotate,
                show_state_after_annotate,
            )
        if method == "markdown":
            return annotate_markdown(
                managers,
                fn_args["kg"],
                fn_args["text_to_be_annotated"],
                state,
                known,
                know_before_annotate,
                show_state_after_annotate,
            )
        raise ValueError(
            f"annotation method '{method}' needs to be one of: matching, prefix, indices, markdown"
        )

    if fn_name == "delete_annotation":
        if method == "matching":
            return delete_annotation_matching(
                fn_args["words_to_be_annotated"],
                fn_args["occurrence_index"],
                state,
                show_state_after_annotate,
            )
        if method == "prefix":
            return annotate_prefix(
                fn_args["optional_short_prefix"],
                fn_args["exact_words_to_be_annotated"],
                fn_args["optional_short_suffix"],
                state,
                show_state_after_annotate,
            )
        if method == "indices":
            return annotate_indices(
                fn_args["start_index"],
                fn_args["end_index"],
                state,
                show_state_after_annotate,
            )
        if method == "markdown":
            raise RuntimeError(
                "markdown annotation method doesn't have a delete function"
            )
        raise ValueError(
            f"annotation method '{method}' needs to be one of: matching, prefix, indices, markdown"
        )

    if fn_name == "show_current_annotations":
        return state.format(only_current_window=True, list_entities=True)

    if fn_name == "finalize":
        return "Finalized the annotation process."

    raise ValueError(f"Unknown function '{fn_name}'")


def feedback_system_message(
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
) -> str:
    return "\n\n".join(
        [
            "You are a text annotation assistant providing feedback on the "
            "output of a text annotation system for a given input text.",
            format_section(
                "Available knowledge graphs",
                format_kgs(managers, kg_notes),
            ),
            format_section(
                "General notes across knowledge graphs",
                format_notes(notes, enumerated=True),
            ),
            format_section(
                "Rules to follow",
                format_enumerate(rules()) if rules() else "None",
            ),
            "Provide your feedback with the give_feedback function.",
        ]
    )


def feedback_instructions(inputs: list[str], output: dict) -> str:
    assert inputs, "At least one input is required for feedback"

    sections = []
    if len(inputs) > 1:
        sections.append(
            format_section(
                "Previous inputs",
                "\n\n".join(i.strip() for i in inputs[:-1]),
            )
        )

    sections.append(format_section("Input", inputs[-1].strip()))
    sections.append(format_section("Annotations", output["formatted"]))
    return "\n\n".join(sections)


class EntityLinkingTask(GraspTask, FeedbackTask):
    name = "entity-linking"

    def system_information(self) -> str:
        return system_information()

    def rules(self) -> list[str]:
        return rules()

    def function_definitions(self) -> list[dict]:
        return functions(self.managers, self.config)

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None = None,
    ) -> str:
        return call_function(
            self.config,
            self.managers,
            fn_name,
            fn_args,
            known,
            self.state,
            example_indices,
        )

    def done(self, fn_name: str) -> bool:
        return fn_name == "finalize"

    def setup(self, input: Any) -> str:
        instructions, self.state = input_and_state(input, self.config)
        return instructions

    def output(self, messages: list[Message]) -> dict:
        return self.state.to_dict()

    @property
    def default_input_field(self) -> str | None:
        return "text"

    @classmethod
    def sample_cls(cls) -> type[EntityLinkingSample] | None:
        return EntityLinkingSample

    def feedback_system_message(
        self, kg_notes: dict[str, list[str]], notes: list[str]
    ) -> str:
        return feedback_system_message(self.managers, kg_notes, notes)

    def feedback_instructions(self, inputs: list[str], output: dict) -> str:
        return feedback_instructions(inputs, output)

