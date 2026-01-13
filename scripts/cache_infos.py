import argparse
import dbm
import json
import os
from itertools import batched

from tqdm import trange
from universal_ml_utils.logging import setup_logging

from grasp.configs import KgConfig
from grasp.manager import load_kg_manager
from grasp.manager.utils import load_data
from grasp.utils import get_available_knowledge_graphs, get_index_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache information from RDF data.")
    parser.add_argument("kg", type=str, choices=get_available_knowledge_graphs())
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=100,
        help="Number of items to process in each batch.",
    )
    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        default=None,
        help="SPARQL endpoint to use for querying the knowledge graph.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10_000_000,
        help="Only cache the top N item",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing cached data.",
    )
    return parser.parse_args()


def cache(args: argparse.Namespace) -> None:
    setup_logging("INFO")

    manager = load_kg_manager(KgConfig(kg=args.kg, endpoint=args.endpoint))

    kg_index_dir = get_index_dir(args.kg)

    info_sparqls = {
        "entities": manager.entity_info_sparql,
        "properties": manager.property_info_sparql,
    }

    for typ, info_sparql in info_sparqls.items():
        sub_dir = os.path.join(kg_index_dir, typ)

        cache_path = os.path.join(sub_dir, "info.cache", "db")
        os.makedirs(cache_path, exist_ok=True)

        data = load_data(sub_dir)

        dbm_mode = "n" if args.overwrite else "c"
        with dbm.open(cache_path, dbm_mode) as db:
            for ids in batched(
                trange(min(len(data), args.limit), desc=f"Caching infos for {typ}"),
                args.batch_size,
            ):
                identifiers = []
                for id in ids:
                    identifier = data.identifier(id)
                    if identifier is None or identifier in db:
                        continue
                    identifiers.append(identifier)

                if not identifiers:
                    continue

                infos = manager.get_infos_for_identifiers(identifiers, info_sparql)
                for identifier, info in infos.items():
                    db[identifier] = json.dumps(info)


if __name__ == "__main__":
    cache(parse_args())
