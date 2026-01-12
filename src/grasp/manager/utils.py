import os
import time
from pathlib import Path
from typing import Type

from search_rdf import Data, KeywordIndex, EmbeddingIndex
from universal_ml_utils.io import dump_json, load_json
from universal_ml_utils.logging import get_logger

from grasp.manager.normalizer import Normalizer, WikidataPropertyNormalizer
from grasp.sparql.utils import get_endpoint, load_qlever_prefixes
from grasp.utils import get_index_dir


SearchIndex = KeywordIndex | EmbeddingIndex


def load_data(index_dir: str) -> Data:
    try:
        data = Data.load(os.path.join(index_dir, "data"))
    except Exception as e:
        raise ValueError(f"Failed to load index data from {index_dir}") from e

    return data


def load_data_and_normalizer(
    index_dir: str,
    normalizer_cls: Type[Normalizer] | None = None,
) -> tuple[Data, Normalizer]:
    data = load_data(index_dir)

    if normalizer_cls is None:
        normalizer_cls = Normalizer

    return data, normalizer_cls()


def load_index_and_normalizer(
    index_dir: str,
    index_type: str,
    normalizer_cls: Type[Normalizer] | None = None,
) -> tuple[SearchIndex, Normalizer]:
    logger = get_logger("KG INDEX LOADING")
    start = time.perf_counter()

    data, normalizer = load_data_and_normalizer(index_dir, normalizer_cls)
    index_dir = os.path.join(index_dir, index_type)
    load_kwargs = {"data": data, "index_dir": index_dir}

    if index_type == "keyword":
        index_cls = KeywordIndex
    elif index_type == "embedding":
        index_cls = EmbeddingIndex
        load_kwargs["embedding_path"] = os.path.join(
            index_dir, "embeddings.safetensors"
        )
    else:
        raise ValueError(f"Unknown index type {index_type}")

    try:
        index = index_cls.load(**load_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load {index_type} index from {index_dir}") from e

    end = time.perf_counter()

    logger.debug(f"Loading {index_type} index from {index_dir} took {end - start:.2f}s")

    return index, normalizer


def load_entity_index_and_normalizer(
    kg: str,
    index_type: str | None = None,
) -> tuple[SearchIndex, Normalizer]:
    index_dir = os.path.join(get_index_dir(kg), "entities")

    return load_index_and_normalizer(
        index_dir,
        # for entities use keyword index by default
        index_type or "keyword",
    )


def load_property_index_and_normalizer(
    kg: str,
    index_type: str | None = None,
) -> tuple[SearchIndex, Normalizer]:
    index_dir = os.path.join(get_index_dir(kg), "properties")

    mapping_cls = WikidataPropertyNormalizer if kg.startswith("wikidata") else None

    return load_index_and_normalizer(
        index_dir,
        # for properties use embedding index by default
        index_type or "embedding",
        mapping_cls,
    )


def load_kg_prefixes(kg: str, endpoint: str | None = None) -> dict[str, str]:
    kg_index_dir = get_index_dir(kg)
    prefix_file = Path(kg_index_dir, "prefixes.json")
    if prefix_file.exists():
        prefixes = load_json(prefix_file.as_posix())
    else:
        try:
            prefixes = load_qlever_prefixes(endpoint or get_endpoint(kg))
            # save for future use
            dump_json(prefixes, prefix_file.as_posix(), indent=2)
        except Exception:
            prefixes = {}

    common_prefixes = get_common_sparql_prefixes()
    values = set(prefixes.values())

    # add common prefixes that might not be covered by the
    # specified prefixes
    for short, long in common_prefixes.items():
        if short in prefixes or long in values:
            continue

        prefixes[short] = long

    return prefixes


def load_kg_info_sparqls(kg: str) -> tuple[str | None, str | None]:
    kg_index_dir = get_index_dir(kg)
    ent_info_file = Path(kg_index_dir, "entities", "info.sparql")
    prop_info_file = Path(kg_index_dir, "properties", "info.sparql")

    if ent_info_file.exists():
        ent_info = ent_info_file.read_text()
    else:
        ent_info = None

    if prop_info_file.exists():
        prop_info = prop_info_file.read_text()
    else:
        prop_info = None

    return ent_info, prop_info


def load_kg_indices(
    kg: str,
    entities_type: str | None = None,
    properties_type: str | None = None,
) -> tuple[SearchIndex, SearchIndex, Normalizer, Normalizer]:
    ent_index, ent_normalizer = load_entity_index_and_normalizer(
        kg,
        entities_type,
    )

    prop_index, prop_normalizer = load_property_index_and_normalizer(
        kg,
        properties_type,
    )

    return ent_index, prop_index, ent_normalizer, prop_normalizer


def get_common_sparql_prefixes() -> dict[str, str]:
    return {
        "rdf": "<http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "<http://www.w3.org/2000/01/rdf-schema#",
        "owl": "<http://www.w3.org/2002/07/owl#",
        "xsd": "<http://www.w3.org/2001/XMLSchema#",
        "foaf": "<http://xmlns.com/foaf/0.1/",
        "skos": "<http://www.w3.org/2004/02/skos/core#",
        "dct": "<http://purl.org/dc/terms/",
        "dc": "<http://purl.org/dc/elements/1.1/",
        "prov": "<http://www.w3.org/ns/prov#",
        "schema": "<http://schema.org/",
        "geo": "<http://www.opengis.net/ont/geosparql#",
        "geosparql": "<http://www.opengis.net/ont/geosparql#",
        "gn": "<http://www.geonames.org/ontology#",
        "bd": "<http://www.bigdata.com/rdf#",
        "hint": "<http://www.bigdata.com/queryHints#",
        "wikibase": "<http://wikiba.se/ontology#",
        "qb": "<http://purl.org/linked-data/cube#",
        "void": "<http://rdfs.org/ns/void#",
    }


def is_embedding_index(index: SearchIndex) -> bool:
    return index.index_type == "embedding"


def describe_index(index_type: str) -> tuple[str, str]:
    if index_type == "keyword":
        title = "Keyword index"
        desc = "Retrieves items by overlap between query keywords \
and item label words. The query keywords can match item label words exactly or \
as prefixes. No special query operators like AND/OR are supported."

    elif index_type == "embedding":
        title = "Embedding index"
        desc = "Retrieves items by cosine similarity between their \
label embeddings and the query embedding."

    else:
        raise ValueError(f"Unknown index type {index_type}")

    return title, desc
