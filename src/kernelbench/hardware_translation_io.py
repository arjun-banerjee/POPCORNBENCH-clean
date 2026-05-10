"""Helpers for hardware-translation mode: TOML I/O contracts + CUDA oracle modules."""

from __future__ import annotations

import ast
import os

import tomli


def extract_io_contract_src(python_src: str) -> str:
    """
    Strip a KernelBench problem .py down to imports, hyperparameter constants,
    ``get_init_inputs`` / ``get_inputs``, and brief instructions.

    Used to prompt the model without exposing the reference ``Model`` forward.
    """
    tree = ast.parse(python_src)
    header = (
        "# I/O contract only — implement ModelNew with custom CUDA for the target GPU.\n"
        "# Use ``torch.utils.cpp_extension.load_inline`` (or equivalent) for device code.\n"
    )
    import_chunks: list[str] = []
    const_chunks: list[str] = []
    seen_class = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            seen_class = True
            break
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(python_src, node)
            if seg:
                import_chunks.append(seg.strip())
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            seg = ast.get_source_segment(python_src, node)
            if seg:
                const_chunks.append(seg.strip())

    func_chunks: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "get_inputs",
            "get_init_inputs",
        ):
            seg = ast.get_source_segment(python_src, node)
            if seg:
                func_chunks.append(seg.strip())

    if len(func_chunks) < 2:
        raise ValueError(
            "extract_io_contract_src: need both get_inputs and get_init_inputs in source"
        )

    parts: list[str] = [header, ""]
    if import_chunks:
        parts.append("\n".join(import_chunks))
        parts.append("")
    if const_chunks:
        parts.append("\n".join(const_chunks))
        parts.append("")
    parts.append("\n\n".join(func_chunks))
    parts.append("")
    parts.append(
        "# ModelNew must define __init__(*get_init_inputs()) and forward(*get_inputs()) "
        "so outputs match the hidden reference implementation on the evaluation GPU."
    )
    if not seen_class:
        raise ValueError("extract_io_contract_src: expected a Model class in problem source")
    return "\n".join(parts)


def load_io_contract_from_toml(
    *,
    repo_top: str,
    io_dir: str,
    problem_name: str,
) -> str:
    """
    Load the pre-rendered tensor / RNG I/O contract for the LLM prompt.

    ``io_dir`` is relative to ``repo_top`` and must contain ``{stem}.toml`` with
    a ``contract = '''...'''`` string (see ``level5/hardware_translation/io``).
    """
    if not io_dir or not str(io_dir).strip():
        raise ValueError(
            "hardware_translation requires hardware_translation_io_dir "
            "(directory of per-problem .toml files with a `contract` field)."
        )

    rel = io_dir.strip()
    path_dir = rel if os.path.isabs(rel) else os.path.join(repo_top, rel)
    stem, _ = os.path.splitext(problem_name)
    path = os.path.join(path_dir, f"{stem}.toml")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"I/O contract TOML not found: {path}")
    with open(path, "rb") as f:
        data = tomli.load(f)
    contract = data.get("contract")
    if not contract or not str(contract).strip():
        raise ValueError(f"TOML at {path} must contain a non-empty `contract` string")
    return str(contract).strip()


def load_oracle_reference_source(
    *,
    repo_top: str,
    oracle_dir: str,
    problem_name: str,
) -> str:
    """
    Load the full Python module used as the numerical oracle in
    ``eval_kernel_against_ref`` (``Model`` + ``get_inputs`` + ``get_init_inputs``).

    Replace ``Model.forward`` with ``load_inline`` of **self-contained** H100 CUDA
    when you are ready; raw ``kernels/h100/*.cu`` snippets usually need headers
    before they compile inside ``load_inline``.
    """
    if not oracle_dir or not str(oracle_dir).strip():
        raise ValueError(
            "hardware_translation requires hardware_translation_oracle_dir "
            "(directory of KernelBench-style oracle .py modules for the target GPU)."
        )

    rel = oracle_dir.strip()
    ref_dir = rel if os.path.isabs(rel) else os.path.join(repo_top, rel)
    stem, _ = os.path.splitext(problem_name)
    candidates = [
        os.path.join(ref_dir, problem_name),
        os.path.join(ref_dir, f"{stem}_oracle.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return f.read()

    raise FileNotFoundError(
        f"Oracle module not found for {problem_name!r} under {ref_dir}. "
        f"Tried {problem_name!r} and {stem}_oracle.py."
    )


def load_reference_model_src_for_eval(
    *,
    repo_top: str,
    problem_name: str,
    reference_kernel_dir: str | None,
) -> str:
    """Deprecated alias for tooling; use ``load_oracle_reference_source``."""
    return load_oracle_reference_source(
        repo_top=repo_top,
        oracle_dir=reference_kernel_dir or "",
        problem_name=problem_name,
    )
