# Compatibility Specification

## Target ROSA Model Parameters

| Property    | BlinkDL target |    Adapter |
| ----------- | -------------: | ---------: |
| Layers      |             12 |         12 |
| Hidden size |            768 |        768 |
| ROSA bits   |              4 |          4 |
| ROSA groups |            192 |        192 |
| Context     |            512 |        512 |
| Matching    | longest suffix | must match |
| Tie-break   |         latest | must match |
| Unmatched   |           zero | must match |

## Output Semantics
- **Matched bit = 1**: `+1.0` (or `+emb`)
- **Matched bit = 0**: `-1.0` (or `-emb`)
- **Unmatched route**: `0.0`

## Upstream Provenance

- **RWKV-LM Repository**: `https://github.com/BlinkDL/RWKV-LM.git`
  - Checked-out Commit SHA: `ec56ea2b172c065a793d25723bc03e2af1f018dd`
  - Targeted Model Script: `RWKV-v8/260222_rosa4bitLM_L12.py`
  - Date Added: `2026-02-22`

- **rosa_soft Repository**: `https://github.com/wjie98/rosa_soft.git`
  - Checked-out Commit SHA: `5cb789872da721455f942218c003529230c67f0a`
  - Date Added: `2026-02-22`

## Suffix Horizon Note
`rosa_soft` defaults to a suffix horizon of 32. Our target model script requires an explicit horizon of 512 (`max_suffix_length=512`), which is enforced by `rosa_compute.rosa_compat`.
