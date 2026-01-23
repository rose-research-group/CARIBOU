METADATA_PROMPT = """
You are given an anonymized single-cell RNA-seq dataset at /workspace/dataset.h5ad.
Your task is to infer dataset-level metadata and write a JSON report.

You must determine:
1) Species
2) Organ
3) Cell count
4) Mean transcript count per cell

Rules:
- Do not assume any hidden metadata exists in obs or uns.
- Use the expression matrix and gene names to infer species and organ.
- Use counts if available (adata.layers["counts"]), otherwise use adata.X.
- Write a JSON file to /workspace/outputs/metadata_inference.json.
- The JSON must include the required fields and nothing confidential.

Required JSON format:
{
  "species": "<string>",
  "organ": "<string>",
  "cell_count": <int>,
  "mean_transcript_count": <float>,
  "confidence": {
    "species": <float 0-1>,
    "organ": <float 0-1>
  },
  "evidence": {
    "species": ["<short reasons>"],
    "organ": ["<short reasons>"]
  }
}

Be concise and include the JSON file write in your code.
"""

FULL_SYSTEM_METADATA_PROMPT = """
You are running in a multi-agent system. Coordinate to complete the task.

Task:
Infer dataset-level metadata from the anonymized single-cell RNA-seq dataset at /workspace/dataset.h5ad.

You must determine:
1) Species
2) Organ
3) Cell count
4) Mean transcript count per cell

Guidance:
- Use specialist agents where helpful (e.g., species inference, organ markers, QC/counts).
- Do not assume metadata exists in obs or uns; the dataset is anonymized.
- Use counts if available (adata.layers["counts"]), otherwise use adata.X.
- Provide evidence: list a few marker genes or gene naming patterns that justify species/organ.

Output requirement:
- Write /workspace/outputs/metadata_inference.json with this exact structure:
{
  "species": "<string>",
  "organ": "<string>",
  "cell_count": <int>,
  "mean_transcript_count": <float>,
  "confidence": {
    "species": <float 0-1>,
    "organ": <float 0-1>
  },
  "evidence": {
    "species": ["<short reasons>"],
    "organ": ["<short reasons>"]
  }
}

Be concise and ensure the JSON file is written before ending the session.
"""
