"""
with_retrieval.py -- scope-based filtering and hybrid retrieval.

A research assistant serves a scientist whose memories span three domains.
Two selection mechanisms are shown:

  Scope filtering: get_policy(scopes=[...]) restricts which memories are
  visible. Fast, deterministic, no embeddings needed.

  Hybrid retrieval: with an embedder configured, get_policy(context=...)
  reranks memories within the selected set by BM25 + vector similarity.
  The highest-relevance memories surface first.

Both can be combined: pass both context= and scopes= to filter to a domain
and then rerank by query relevance within it.

Requirements:
  pip install imprint-mem[vector,openai]
  export ANTHROPIC_API_KEY=sk-ant-...
  export OPENAI_API_KEY=sk-...

  # Voyage alternative: pip install imprint-mem[vector,voyage]
  # Swap OpenAIEmbedder for VoyageEmbedder -- no OPENAI_API_KEY needed.

Usage:
  python examples/with_retrieval.py
"""

import asyncio
from pathlib import Path

from imprint import Imprint, SQLiteMemoryStore, SQLiteVecStore
from imprint.providers.openai import OpenAIEmbedder

DB_PATH = "research_assistant.db"
DIM = 512


async def main() -> None:
    Path(DB_PATH).unlink(missing_ok=True)

    store = SQLiteMemoryStore(DB_PATH)
    await store.connect()

    embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=DIM)

    imprint = Imprint(
        agent_id="research_assistant",
        store=store,
        vector_store=SQLiteVecStore(store.conn, dim=DIM),
        embedder=embedder,
        processing_mode="frugal",
        scopes=["domain:biology", "domain:chemistry", "domain:materials"],
    )
    await imprint.connect()

    print("=== Research Assistant (scope filtering + hybrid retrieval) ===\n")

    # ------------------------------------------------------------------
    # Seed memories with domain-specific vocabulary so both BM25 and
    # vector similarity can discriminate between domains.
    # Embeddings are stored transparently at write time.
    # ------------------------------------------------------------------

    await imprint.observe_directions(
        user_id="sam",
        scope="domain:biology",
        directions=[
            "When reporting enzyme kinetics, always include Km and Vmax "
            "with error bars from triplicate experiments.",
            "Specify cell line, passage number, and serum lot for all "
            "mammalian cell culture experiments.",
            "Use molarity (mol/L) for buffer concentrations in biochemistry "
            "protocols, never mg/mL.",
            "Quantify Western blots normalized to total protein loading, "
            "not just a housekeeping gene.",
            "Cite primary biochemistry journals (JBC, PNAS, Nat Methods) "
            "over review articles or textbooks.",
        ],
    )

    await imprint.observe_directions(
        user_id="sam",
        scope="domain:chemistry",
        directions=[
            "Report synthesis yields as isolated yield after chromatographic "
            "purification, never crude NMR yield.",
            "For palladium-catalyzed reactions, always specify catalyst loading, "
            "ligand identity, and base as mol%.",
            "NMR spectra must state field strength in MHz, deuterated solvent, "
            "and chemical shift referencing.",
            "Use IUPAC systematic nomenclature for all organic compounds on first mention.",
            "Express reaction stoichiometry as equivalents relative to the limiting reagent.",
        ],
    )

    await imprint.observe_directions(
        user_id="sam",
        scope="domain:materials",
        directions=[
            "Report thin film thickness from profilometry or ellipsometry "
            "measurements, always include measurement error.",
            "Document substrate pre-treatment (plasma cleaning, UV-ozone) "
            "before every deposition step.",
            "Characterize microstructure by XRD with Cu Kalpha radiation "
            "and report peak positions and FWHM.",
            "Report all mechanical properties in SI units: GPa for modulus, "
            "MPa for yield strength.",
        ],
    )

    total = await imprint.list_memories("sam")
    print(f"Seeded {len(total)} memories across 3 domains.\n")

    # ------------------------------------------------------------------
    # Mechanism 1: Explicit scope filtering.
    # get_policy(scopes=[...]) restricts which memories are visible.
    # Deterministic -- no embeddings needed, no LLM inference.
    # ------------------------------------------------------------------

    print("--- Scope filtering (explicit) ---\n")

    for scope, label in [
        ("domain:biology", "biology"),
        ("domain:chemistry", "chemistry"),
        ("domain:materials", "materials"),
    ]:
        p = await imprint.get_policy(user_id="sam", scopes=[scope])
        print(f"{label} scope -> {len(p.memories)} memories:")
        for m in p.memories:
            print(f"  {m.content[:70]}")
        print()

    # ------------------------------------------------------------------
    # Mechanism 2: Scope filtering + hybrid reranking within the scope.
    # Adding context= reranks the filtered set by BM25 + vector similarity.
    # The memory most relevant to the query surfaces first.
    # ------------------------------------------------------------------

    print("--- Scope filtering + hybrid reranking (context within scope) ---\n")

    p_bio = await imprint.get_policy(
        user_id="sam",
        scopes=["domain:biology"],
        context="reporting Michaelis-Menten kinetics and Km values for an enzyme assay",
    )
    print(f"Biology scope + enzyme kinetics context -> {len(p_bio.memories)} memories:")
    for m in p_bio.memories:
        print(f"  {m.content[:70]}")

    print()

    p_chem = await imprint.get_policy(
        user_id="sam",
        scopes=["domain:chemistry"],
        context="checking palladium catalyst loading and NMR yield for a coupling reaction",
    )
    print(f"Chemistry scope + synthesis context -> {len(p_chem.memories)} memories:")
    for m in p_chem.memories:
        print(f"  {m.content[:70]}")

    await imprint.close()
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
    Path(DB_PATH).unlink(missing_ok=True)
