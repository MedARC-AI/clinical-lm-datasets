# MIMIC-III

| Category          | Counts | Include  | Comments                                          |
| ----------------- | ------ | -------- | ------------------------------------------------- |
| Nursing/other     | 82,2497 | No      | Choppy, should be in dsum                         |
| Radiology         | 522,279 | **Yes** | Use Impressions and Findings (downsample?)        |
| Nursing           | 223,556 | No      | See above                                         |
| ECG               | 209,051 | No      | Very repetitive and short                         |
| Physician         | 141,624 | No      | Seem like Progress Notes. Lot of lab values.      |
| Discharge Summary | 59,652  | **Yes** | Summary of previous notes.                        |
| Echo              | 45,794  | **Yes** | Repetitive? Take everything after INTERPRETATION: |
| Respiratory       | 31,739  | No      | Repetitive, no sentences.                         |
| Nutrition         | 9,418   | No      | Mostly lab values.                                |
| General           | 8,301   | **Yes** | Admit / Progress notes                            |
| Rehab Services    | 5,431   | No      | PT notes. Not many full sentences.                |
| Social Work       | 2,670   | No      | Does not address patient hospital course much     |
| Case Management   | 967     | No      | Does not address patient hospital course much     |
| Pharmacy          | 103     | **Yes** | Helpful med information (dosages, indications)    |
| Consult           | 98      | **Yes** | Quality (dedup w dsum)                            |

# MIMIC-IV
