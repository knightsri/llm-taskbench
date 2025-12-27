"""
USE-CASE.md Parser and Analyzer.

This module parses human-friendly Markdown use case definitions and analyzes
the data/ground-truth folders to derive evaluation criteria and prompts.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DataFile:
    """Represents an input data file."""
    path: Path
    name: str
    extension: str
    content_preview: str = ""
    size_bytes: int = 0
    line_count: int = 0


@dataclass
class GroundTruthFile:
    """Represents an expected output file."""
    path: Path
    name: str
    extension: str
    content: Any = None  # Parsed content (JSON, CSV rows, etc.)
    format_type: str = "unknown"  # json, csv, text


@dataclass
class DataGroundTruthPair:
    """A matched pair of input data and expected output."""
    data_file: DataFile
    ground_truth_file: GroundTruthFile
    match_pattern: str  # How they were matched (e.g., "lecture-01")


@dataclass
class ParsedUseCase:
    """Structured representation of a parsed USE-CASE.md."""
    folder_path: Path
    name: str = ""
    goal: str = ""
    difficulty: str = ""
    primary_capability: str = ""
    token_range: str = ""
    llm_notes: str = ""
    expected_output_schema: str = ""
    output_format: str = "json"  # json, csv, text
    quality_criteria: str = ""
    edge_cases: List[str] = field(default_factory=list)
    evaluation_script: str = ""

    # Derived from data/ground-truth analysis
    data_files: List[DataFile] = field(default_factory=list)
    ground_truth_files: List[GroundTruthFile] = field(default_factory=list)
    matched_pairs: List[DataGroundTruthPair] = field(default_factory=list)

    # Generated prompts (saved to use case folder)
    prompts: Dict[str, str] = field(default_factory=dict)


class UseCaseMarkdownParser:
    """
    Parses USE-CASE.md files and extracts structured information.
    """

    # Section patterns to extract
    SECTION_PATTERNS = {
        "metadata": r"## Metadata\s*\n(.*?)(?=\n## |\Z)",
        "goal": r"## Goal\s*\n(.*?)(?=\n## |\Z)",
        "llm_notes": r"## LLM Evaluation Notes\s*\n(.*?)(?=\n## |\Z)",
        "expected_output": r"## Expected Output Schema\s*\n(.*?)(?=\n## |\Z)",
        "quality_criteria": r"## Quality Criteria\s*\n(.*?)(?=\n## |\Z)",
        "duration_rules": r"## Duration Rules\s*\n(.*?)(?=\n## |\Z)",
        "evaluation_script": r"## Evaluation Script\s*\n(.*?)(?=\n## |\Z)",
    }

    def parse(self, usecase_md_path: Path) -> ParsedUseCase:
        """
        Parse a USE-CASE.md file into structured data.

        Args:
            usecase_md_path: Path to the USE-CASE.md file

        Returns:
            ParsedUseCase with extracted fields
        """
        content = usecase_md_path.read_text(encoding="utf-8")
        folder_path = usecase_md_path.parent

        parsed = ParsedUseCase(folder_path=folder_path)

        # Extract title (first # heading)
        title_match = re.search(r"^# (?:Use Case: )?(.+)$", content, re.MULTILINE)
        if title_match:
            parsed.name = title_match.group(1).strip()

        # Extract metadata section
        metadata = self._extract_section(content, "metadata")
        if metadata:
            parsed.difficulty = self._extract_metadata_field(metadata, "Difficulty")
            parsed.primary_capability = self._extract_metadata_field(metadata, "Primary Capability")
            parsed.token_range = self._extract_metadata_field(metadata, "Token Range")
            # Try to extract name from metadata if not found in title
            if not parsed.name:
                parsed.name = self._extract_metadata_field(metadata, "Name")

        # Extract goal
        parsed.goal = self._extract_section(content, "goal") or ""

        # Extract LLM notes
        parsed.llm_notes = self._extract_section(content, "llm_notes") or ""

        # Extract expected output schema
        output_section = self._extract_section(content, "expected_output") or ""
        parsed.expected_output_schema = output_section

        # Detect output format from schema
        if "```csv" in output_section.lower() or "csv" in output_section.lower():
            parsed.output_format = "csv"
        elif "```json" in output_section.lower():
            parsed.output_format = "json"

        # Extract quality criteria
        parsed.quality_criteria = self._extract_section(content, "quality_criteria") or ""

        # Extract edge cases from LLM notes
        parsed.edge_cases = self._extract_edge_cases(content)

        # Extract evaluation script
        eval_section = self._extract_section(content, "evaluation_script") or ""
        if eval_section:
            # Extract Python code block
            code_match = re.search(r"```python\s*\n(.*?)```", eval_section, re.DOTALL)
            if code_match:
                parsed.evaluation_script = code_match.group(1).strip()

        return parsed

    def _extract_section(self, content: str, section_key: str) -> Optional[str]:
        """Extract a section from the markdown content."""
        pattern = self.SECTION_PATTERNS.get(section_key)
        if not pattern:
            return None

        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_metadata_field(self, metadata: str, field_name: str) -> str:
        """Extract a specific field from the metadata section."""
        pattern = rf"\*\*{field_name}:\*\*\s*(.+?)(?:\n|$)"
        match = re.search(pattern, metadata)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_edge_cases(self, content: str) -> List[str]:
        """Extract edge cases from the content."""
        edge_cases = []

        # Look for "Edge cases to watch:" or similar
        pattern = r"\*\*Edge cases to watch:\*\*\s*\n((?:- .+\n?)+)"
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            cases_text = match.group(1)
            for line in cases_text.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    edge_cases.append(line[2:])

        return edge_cases


class DataGroundTruthAnalyzer:
    """
    Analyzes data/ and ground-truth/ folders to understand input-output relationships.
    """

    def analyze(self, usecase_folder: Path) -> Tuple[List[DataFile], List[GroundTruthFile], List[DataGroundTruthPair]]:
        """
        Analyze the data and ground-truth folders.

        Args:
            usecase_folder: Path to the use case folder

        Returns:
            Tuple of (data_files, ground_truth_files, matched_pairs)
        """
        data_folder = usecase_folder / "data"
        gt_folder = usecase_folder / "ground-truth"

        data_files = self._scan_data_folder(data_folder)
        gt_files = self._scan_ground_truth_folder(gt_folder)
        matched_pairs = self._match_data_to_ground_truth(data_files, gt_files)

        return data_files, gt_files, matched_pairs

    def _scan_data_folder(self, data_folder: Path) -> List[DataFile]:
        """Scan the data folder for input files."""
        data_files = []

        if not data_folder.exists():
            logger.warning(f"Data folder not found: {data_folder}")
            return data_files

        for file_path in data_folder.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                df = DataFile(
                    path=file_path,
                    name=file_path.stem,
                    extension=file_path.suffix.lower(),
                    size_bytes=file_path.stat().st_size
                )

                # Get preview and line count
                try:
                    content = file_path.read_text(encoding="utf-8")
                    df.content_preview = content[:1000]
                    df.line_count = len(content.split("\n"))
                except Exception as e:
                    logger.warning(f"Could not read {file_path}: {e}")

                data_files.append(df)

        return sorted(data_files, key=lambda x: x.name)

    def _scan_ground_truth_folder(self, gt_folder: Path) -> List[GroundTruthFile]:
        """Scan the ground-truth folder for expected output files."""
        gt_files = []

        if not gt_folder.exists():
            logger.warning(f"Ground-truth folder not found: {gt_folder}")
            return gt_files

        for file_path in gt_folder.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                gtf = GroundTruthFile(
                    path=file_path,
                    name=file_path.stem,
                    extension=file_path.suffix.lower()
                )

                # Parse content based on extension
                try:
                    if gtf.extension == ".json":
                        gtf.content = json.loads(file_path.read_text(encoding="utf-8"))
                        gtf.format_type = "json"
                    elif gtf.extension == ".csv":
                        gtf.content = file_path.read_text(encoding="utf-8")
                        gtf.format_type = "csv"
                    else:
                        gtf.content = file_path.read_text(encoding="utf-8")
                        gtf.format_type = "text"
                except Exception as e:
                    logger.warning(f"Could not parse {file_path}: {e}")

                gt_files.append(gtf)

        return sorted(gt_files, key=lambda x: x.name)

    def _match_data_to_ground_truth(
        self,
        data_files: List[DataFile],
        gt_files: List[GroundTruthFile]
    ) -> List[DataGroundTruthPair]:
        """
        Match input data files to their expected output files.

        Uses naming pattern matching:
        - data/lecture-01-python.txt -> ground-truth/lecture-01-*.json
        - data/meeting-02-planning.txt -> ground-truth/meeting-02-*.json
        """
        pairs = []

        for data_file in data_files:
            # Extract the base pattern (e.g., "lecture-01" from "lecture-01-python")
            # Try multiple patterns
            base_patterns = self._extract_base_patterns(data_file.name)

            for pattern in base_patterns:
                # Find matching ground truth files
                matching_gt = [
                    gt for gt in gt_files
                    if gt.name.startswith(pattern) or pattern in gt.name
                ]

                if matching_gt:
                    # Use the first match (or primary match)
                    # Prefer exact prefix match
                    primary_gt = matching_gt[0]
                    for gt in matching_gt:
                        if gt.name.startswith(pattern):
                            primary_gt = gt
                            break

                    pairs.append(DataGroundTruthPair(
                        data_file=data_file,
                        ground_truth_file=primary_gt,
                        match_pattern=pattern
                    ))
                    break

        return pairs

    def _extract_base_patterns(self, filename: str) -> List[str]:
        """Extract potential base patterns from a filename."""
        patterns = []

        # Pattern 1: First two parts separated by hyphen (lecture-01, meeting-02)
        parts = filename.split("-")
        if len(parts) >= 2:
            patterns.append(f"{parts[0]}-{parts[1]}")

        # Pattern 2: Everything before the last hyphen
        if "-" in filename:
            patterns.append(filename.rsplit("-", 1)[0])

        # Pattern 3: Full filename
        patterns.append(filename)

        return patterns


def load_usecase_from_folder(folder_path: str) -> ParsedUseCase:
    """
    Load and analyze a use case from a folder.

    Args:
        folder_path: Path to the use case folder (e.g., "sample-usecases/00-lecture-concept-extraction")

    Returns:
        ParsedUseCase with all extracted and analyzed information
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Use case folder not found: {folder_path}")

    # Find USE-CASE.md
    usecase_md = folder / "USE-CASE.md"
    if not usecase_md.exists():
        raise FileNotFoundError(f"USE-CASE.md not found in: {folder_path}")

    # Parse the markdown
    parser = UseCaseMarkdownParser()
    parsed = parser.parse(usecase_md)

    # Analyze data and ground-truth
    analyzer = DataGroundTruthAnalyzer()
    data_files, gt_files, matched_pairs = analyzer.analyze(folder)

    parsed.data_files = data_files
    parsed.ground_truth_files = gt_files
    parsed.matched_pairs = matched_pairs

    logger.info(
        f"Loaded use case '{parsed.name}' with {len(data_files)} data files, "
        f"{len(gt_files)} ground-truth files, {len(matched_pairs)} matched pairs"
    )

    return parsed


def list_sample_usecases(base_folder: str = "sample-usecases") -> List[Dict[str, Any]]:
    """
    List all available use cases in the sample-usecases folder.

    Returns:
        List of dicts with use case info (path, name, difficulty, etc.)
    """
    base = Path(base_folder)
    usecases = []

    if not base.exists():
        logger.warning(f"Sample usecases folder not found: {base_folder}")
        return usecases

    for item in sorted(base.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            usecase_md = item / "USE-CASE.md"
            if usecase_md.exists():
                try:
                    parsed = load_usecase_from_folder(str(item))
                    usecases.append({
                        "path": str(item),
                        "folder_name": item.name,
                        "name": parsed.name,
                        "difficulty": parsed.difficulty,
                        "capability": parsed.primary_capability,
                        "data_files": len(parsed.data_files),
                        "ground_truth_files": len(parsed.ground_truth_files),
                        "matched_pairs": len(parsed.matched_pairs),
                        "output_format": parsed.output_format
                    })
                except Exception as e:
                    logger.warning(f"Could not load use case from {item}: {e}")
                    usecases.append({
                        "path": str(item),
                        "folder_name": item.name,
                        "name": item.name,
                        "error": str(e)
                    })

    return usecases
