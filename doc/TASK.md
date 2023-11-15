# Problem #

Understanding of how RNA molecules fold.

# Goal #

Prediction the reactivity (DMS, and 2A3) at every position for a particular RNA sequence.

# Model #

Regressor of two values:

- reactivity_DMS_MaP:
  A float value from 0-1 measuring reactivity to the DMS chemical, with 1 being the most reactive and 0 being the least
  reactive.
- reactivity_2A3_MaP:
  A float value from 0-1 measuring reactivity to the 2A3 chemical, with 1 being the most reactive and 0 being the least
  reactive.

# Train Data #

- sequence_id: An arbitrary identifier for each RNA sequence.
- **sequence**: Describes the RNA sequence as a string of `A`, `C`, `G`, and `U`.
- **experiment_type**: Specifies whether the experiment used `DMS_MaP` or `2A3_MaP` for chemical mapping.
- dataset_name: The name of the high throughput
  sequencing [test_sequences.csv](..%2F..%2F..%2Fdata%2Ftest_sequences.csv)dataset from which the reactivity profile was
  extracted.
- reads: Number of reads in the high throughput sequencing experiment assigned to the RNA sequence.
    - The number of reads is important because it can provide information about the abundance of the RNA sequence in the
      sample. Generally, a higher number of reads for an RNA sequence suggests that it is more abundant in the sample.
- signal_to_noise: A signal-to-noise value for the profile, calculated as the mean of measurement values over probed
  nucleotides divided by the mean of statistical errors in measurement values over probed nucleotides.
- SN_filter: A boolean indicating whether the profile passes a signal-to-noise threshold.
- reactivity_0001, reactivity_0002, ...: An array of floating-point numbers that define the reactivity profile for the
  RNA sequence.
- reactivity_error_0001, reactivity_error_0002, ...: Corresponding errors in experimental values obtained in reactivity.

# Train Data Statistics #

|                  Title                   |                 Statistics                  |                                                                                               Code                                                                                                |
|:----------------------------------------:|:-------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|            Train Data Samples            |                  1,643,680                  |                                                                                   `SELECT COUNT(*) FROM train;`                                                                                   | 
|               Experiments                | `2A3_MaP`: 821,840 <br/> `DMS_MaP`: 821,840 |                                                              `SELECT experiment_type, COUNT(*) FROM train GROUP BY experiment_type;`                                                              | 
|           Unique Sequence IDs            |                   806,573                   |                                                           `SELECT COUNT(*) FROM (SELECT sequence_id FROM train GROUP BY sequence_id);`                                                            |
| Samples With More Than Twice Occurrences |                   40,348                    |   `SELECT SUM(occurance_count) AS total_occurances FROM (SELECT sequence_id, COUNT(sequence_id) AS occurance_count FROM train GROUP BY sequence_id HAVING COUNT(sequence_id) > 2) AS subquery;`   |
|   Samples With Additional Occurrences    |                   30,534                    | `SELECT SUM(occurance_count - 2) AS total_occurances FROM (SELECT sequence_id, COUNT(sequence_id) AS occurance_count FROM train GROUP BY sequence_id HAVING COUNT(sequence_id) > 2) AS subquery;` |
|    Samples with NaN reactivity vector    |                   81,480                    |                                                                               [NaN_Samples](../note/nan_samples.py)                                                                               |

| Sequence Length | Occurrences |                                                 Code                                                  |
|:---------------:|:-----------:|:-----------------------------------------------------------------------------------------------------:|
|       115       |    27290    | `SELECT LENGTH(train.sequence) AS seq_length, COUNT(*) as occurances FROM train GROUP BY seq_length;` |
|       177       |   1568354   |                                                                                                       |
|       155       |    13038    |                                                                                                       |
|       206       |    4998     |                                                                                                       |
|       170       |    30000    |                                                                                                       |

# Test Data Statistics #

| Sequence Length | Occurrences |                                                    Code                                                    |
|:---------------:|:-----------:|:----------------------------------------------------------------------------------------------------------:|
|       177       |   335823    | `Select (id_max - id_min + 1) AS length, COUNT(*) AS occurances FROM test GROUP BY length ORDER BY length` |
|       207       |   1000000   |                                                                                                            |
|       307       |    2000     |                                                                                                            |
|       457       |    6000     |                                                                                                            |
