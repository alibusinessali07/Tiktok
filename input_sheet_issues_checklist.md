# Input Sheet Issues Checklist

## 1. Tab Name Typo

- Rename tab `Febraury 2026` to `February 2026`.

## 2. Column Schema Inconsistency (Main Issue)

- `Febraury 2026`: 13 columns (includes `Type` and `Genre`).
- `January 2026` / `December 2025`: 12 columns (missing `Type`).
- `November 2025` to `June 2025`: 11 columns (missing `Type` and `Genre`).
- Because of this, column positions shift between tabs.

## 3. Header Name Inconsistency

- Column A header varies:
  - `Name`
  - `Editor Name`
  - `tertuzz` (September tab)
- `Payment Status` sometimes has leading space:
  -  `Payment Status` (July/June)
- `# edits / # views` vs `# edits`  (June)
- `Campaing Title` typo appears (should be `Campaign Title` if used).

## 4. Payment Status / Date Alignment Risk

- Due to shifted schemas, `Payment Status` and `Payment date` can be read from wrong columns.
- Standardize both fields to the same exact columns in every tab.

## 5. Invalid Data Observed in Paid Rows

- 3 rows had unusable `Payment date`.
- 189 paid rows had invalid/non-canonical TikTok profile values.
- These are dropped by script validation.

## 6. Month Resolution Reality

- In the latest run, all kept paid rows resolved to `February 2026`.
- If older tabs should count for earlier months, `Payment date` values in those rows must actually be earlier months.

## Recommended Unified Column Order (All Tabs)

1. `Name`
2. `Song Title` (or `Campaign Title`, but keep one consistent name)
3. `Type` (keep position consistent if present)
4. `Type of Compensation`
5. `TikTok Profile`
6. `Paypal/Crypto`
7. `Price per edit`
8. `# edits / # views`
9. `Total amount (USD)`
10. `Genre`
11. `Requested by`
12. `Payment Status`
13. `Payment date`

