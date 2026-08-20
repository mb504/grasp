import unittest
from unittest.mock import patch
from entity_linking import (
    annotate_matching,
    annotate_indices,
    annotate_prefix,
    annotate_markdown,
    handle_annotation,
    Text,
    AnnotationState,
)

example_input = """Pat O'Keeffe (1883–1960) was a British professional boxer who \
twice held the British middleweight title. Born in Bromley-by-Bow to Irish parents, he \
turned professional in 1902 and won the middleweight championship of England in 1906. \
Between 1907 and 1910, he fought across the United States and Australia, drawing with \
Billy Papke in Philadelphia and touring with the world heavyweight champion Tommy \
Burns, whom he seconded against Jack Johnson. In 1914, O'Keeffe challenged Georges \
Carpentier in a bout billed as the European heavyweight championship and was knocked \
out in two rounds."""

def fake_find_manager(a: any, b: any):
    return "probably a manager", ""

# helper function to test annotation methods only without other parts of grasp
def fake_handle_annotation(
    manager: any,
    entity: str,
    start_idx: int,
    end_idx: int,
    state: any,
    known: any,
    know_before_annotate: bool,
    show_state_after_annotate: bool,
    add_word_indices: bool = False,
    ):
    return (start_idx, end_idx)

# helper function to test markdown annotation method only without other parts of grasp
def fake_handle_annotation_markdown(
    manager: any,
    entity: str,
    start_idx: int,
    end_idx: int,
    state: any,
    known: any,
    know_before_annotate: bool,
    show_state_after_annotate: bool,
    add_word_indices: bool = False,
    ):
    return f"({start_idx}, {end_idx})"

#######################################################################################
#                            Test prefix annotation method                            #
#######################################################################################

class TestPrefixAnnotationMethod(unittest.TestCase):
    text = Text(data=example_input)
    ann_state = AnnotationState(text)
    # first words of the text to test edge case"
    def test_annotation_at_the_beginning_of_text(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_prefix(
                managers=["manager"],
                kg="WikiFake",
                words_to_be_annotated="Pat O'Keeffe",
                prefix="",
                suffix=" (1883–1960)",
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (0, 12)

    # first occurrence of "O'Keeffe"
    def test_annotation_occurrence_index_0(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_prefix(
                managers=["manager"],
                kg="WikiFake",
                words_to_be_annotated="O'Keeffe",
                prefix="Pat",
                suffix=" (1883–1960)",
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (4, 12)

    # second occurrence of "O'Keeffe"
    def test_annotation_occurrence_index_1(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_prefix(
                managers=["manager"],
                kg="WikiFake",
                words_to_be_annotated="O'Keeffe",
                prefix="",
                suffix="challenged",
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (458, 466)

    # "O'Keeffe" without context is underspecified -> error
    def test_annotation_occurrence_index_too_large(self):
        with (
            self.assertRaises(ValueError),
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            annotate_prefix(
                managers=["manager"],
                kg="WikiFake",
                words_to_be_annotated="O'Keeffe",
                prefix="",
                suffix="",
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            )

    # last word of the text to test edge case"
    def test_annotation_at_the_end_of_text(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_prefix(
                managers=["manager"],
                kg="WikiFake",
                words_to_be_annotated="rounds.",
                prefix="two",
                suffix="",
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (582, 589)


#######################################################################################
#                          Test matching annotation method                            #
#######################################################################################

class TestMatchingAnnotationMethod(unittest.TestCase):
    text = Text(data=example_input)
    ann_state = AnnotationState(text)

    # first words of the text to test edge case"
    def test_annotation_at_the_beginning_of_text(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_matching(
                managers=["manager"],
                kg="WikiFake",
                words_to_be_annotated="Pat O'Keeffe",
                occurrence_index=0,
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (0, 12)

    # first occurrence of "O'Keeffe"
    def test_annotation_occurrence_index_0(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_matching(
                managers=["manager"],
                kg="WikiFake",
                words_to_be_annotated="O'Keeffe",
                occurrence_index=0,
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (4, 12)

    # second occurrence of "O'Keeffe"
    def test_annotation_occurrence_index_1(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_matching(
                managers=["manager"],
                kg="WikiFake",
                words_to_be_annotated="O'Keeffe",
                occurrence_index=1,
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (458, 466)

    # third occurrence of "O'Keeffe" does not exist -> error
    def test_annotation_occurrence_index_too_large(self):
        with (
            self.assertRaises(ValueError),
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            annotate_matching(
                managers=["manager"],
                kg="WikiFake",
                words_to_be_annotated="O'Keeffe",
                occurrence_index=2,
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            )

    # last word of the text to test edge case"
    def test_annotation_at_the_end_of_text(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_matching(
                managers=["manager"],
                kg="WikiFake",
                words_to_be_annotated="rounds.",
                occurrence_index=0,
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (582, 589)


#######################################################################################
#                           Test indices annotation method                            #
#######################################################################################

class TestIndicesAnnotationMethod(unittest.TestCase):
    text = Text(data=example_input)
    ann_state = AnnotationState(text)
    ann_state.format(add_word_indices=True)

    # first words of the text to test edge case"
    def test_annotation_at_the_beginning_of_text(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_indices(
                managers=["manager"],
                kg="WikiFake",
                start=0,
                end=2,
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (0, 12)

    # first occurrence of "O'Keeffe"
    def test_annotation_occurrence_index_0(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_indices(
                managers=["manager"],
                kg="WikiFake",
                start=1,
                end=2,
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (4, 12)

    # second occurrence of "O'Keeffe"
    def test_annotation_occurrence_index_1(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_indices(
                managers=["manager"],
                kg="WikiFake",
                start=74,
                end=75,
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (458, 466)

    # end index too large -> error
    def test_annotation_occurrence_index_too_large(self):
        with (
            self.assertRaises(ValueError),
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_indices(
                managers=["manager"],
                kg="WikiFake",
                start=94,
                end=95,
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            )

    # last word of the text to test edge case"
    def test_annotation_at_the_end_of_text(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation)
            ):
            assert annotate_indices(
                managers=["manager"],
                kg="WikiFake",
                start=94,
                end=94,
                entity="Q123",
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ) == (582, 588)
        # one less than in the other methods, since the full stop at the end does not
        # count as a word in the indices logic


#######################################################################################
#                          Test markdown annotation method                            #
#######################################################################################

class TestMarkdownAnnotationMethod(unittest.TestCase):
    text = Text(data=example_input)
    ann_state = AnnotationState(text)

    # first words of the text to test edge case"
    def test_annotation_at_the_beginning_of_text(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation_markdown)
            ):
            text_to_be_ann =  """⟦Pat O'Keeffe⟧(Q123) (1883–1960) was a British professional boxer who \
twice held the British middleweight title. Born in Bromley-by-Bow to Irish parents, he \
turned professional in 1902 and won the middleweight championship of England in 1906. \
Between 1907 and 1910, he fought across the United States and Australia, drawing with \
Billy Papke in Philadelphia and touring with the world heavyweight champion Tommy \
Burns, whom he seconded against Jack Johnson. In 1914, O'Keeffe challenged Georges \
Carpentier in a bout billed as the European heavyweight championship and was knocked \
out in two rounds."""

            assert annotate_markdown(
                managers=["manager"],
                kg="WikiFake",
                text_to_be_annotated=text_to_be_ann,
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ).startswith("\n- (0, 12)")

    # first occurrence of "O'Keeffe"

    def test_annotation_occurrence_index_0(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation_markdown)
            ):
            text_to_be_ann =  """Pat ⟦O'Keeffe⟧(Q123) (1883–1960) was a British professional boxer who \
twice held the British middleweight title. Born in Bromley-by-Bow to Irish parents, he \
turned professional in 1902 and won the middleweight championship of England in 1906. \
Between 1907 and 1910, he fought across the United States and Australia, drawing with \
Billy Papke in Philadelphia and touring with the world heavyweight champion Tommy \
Burns, whom he seconded against Jack Johnson. In 1914, O'Keeffe challenged Georges \
Carpentier in a bout billed as the European heavyweight championship and was knocked \
out in two rounds."""

            assert annotate_markdown(
                managers=["manager"],
                kg="WikiFake",
                text_to_be_annotated=text_to_be_ann,
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ).startswith("\n- (4, 12)")

    # second occurrence of "O'Keeffe"
    def test_annotation_occurrence_index_1(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation_markdown)
            ):
            text_to_be_ann =  """Pat O'Keeffe (1883–1960) was a British professional boxer who \
twice held the British middleweight title. Born in Bromley-by-Bow to Irish parents, he \
turned professional in 1902 and won the middleweight championship of England in 1906. \
Between 1907 and 1910, he fought across the United States and Australia, drawing with \
Billy Papke in Philadelphia and touring with the world heavyweight champion Tommy \
Burns, whom he seconded against Jack Johnson. In 1914, ⟦O'Keeffe⟧(Q123) challenged Georges \
Carpentier in a bout billed as the European heavyweight championship and was knocked \
out in two rounds."""

            assert annotate_markdown(
                managers=["manager"],
                kg="WikiFake",
                text_to_be_annotated=text_to_be_ann,
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ).startswith("\n- (458, 466)")

    # last word of the text to test edge case"
    def test_annotation_at_the_end_of_text(self):
        with (
            patch("entity_linking.find_manager", fake_find_manager),
            patch("entity_linking.handle_annotation", fake_handle_annotation_markdown)
            ):
            text_to_be_ann =  """Pat O'Keeffe (1883–1960) was a British professional boxer who \
twice held the British middleweight title. Born in Bromley-by-Bow to Irish parents, he \
turned professional in 1902 and won the middleweight championship of England in 1906. \
Between 1907 and 1910, he fought across the United States and Australia, drawing with \
Billy Papke in Philadelphia and touring with the world heavyweight champion Tommy \
Burns, whom he seconded against Jack Johnson. In 1914, O'Keeffe challenged Georges \
Carpentier in a bout billed as the European heavyweight championship and was knocked \
out in two ⟦rounds.⟧(Q123)"""

            assert annotate_markdown(
                managers=["manager"],
                kg="WikiFake",
                text_to_be_annotated=text_to_be_ann,
                state=self.ann_state,
                known=set(),
                know_before_annotate=False,
                show_state_after_annotate=True,
            ).startswith("\n- (582, 589)")


if __name__ == '__main__':
    unittest.main()