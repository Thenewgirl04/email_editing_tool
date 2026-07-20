# Reference Artifacts

These files are preserved evaluation outputs from the earlier `oldwork` branch. They are used by the **Reference Analysis** page for the shorten-task comparison charts.

| File | Description |
| --- | --- |
| `evaluation_shorten_results.csv` | Shorten-task baseline run (50 emails) |
| `evaluation_shorten_with_edgecases_results.csv` | Shorten-task run with edge cases (60 emails) |
| `evaluation_.csv` | Partial export from an earlier run |

## Notes

- These files include legacy `word_count_*` columns from the old evaluation setup. The current pipeline uses faithfulness, completeness, and relevance only.
- Current evaluation output is written to `results/baseline/` and `results/with_edge_cases/` when you run the Evaluation Pipeline page.
