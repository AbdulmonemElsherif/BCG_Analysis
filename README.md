# BCG J-Peak Detection and Heart Rate Estimation

This project adapts the J-Peak detection algorithm from the [CodeOcean capsule (1398208)](https://codeocean.com/capsule/1398208/tree) to work with the ballistocardiogram (BCG) dataset available from [Figshare](https://doi.org/10.6084/m9.figshare.26013157).

## Project Structure

- `adapt_capsule.py`: Original adaptation script (not required to run)
- `process_single_subject.py`: Script to process a single subject
- `process_all_subjects.py`: Script to process all subjects in the dataset
- `visualization.py`: Helper functions for visualization
- `test_sample.py`: Simple test script for basic processing
- `capsule-1398208/`: Original CodeOcean capsule code
- `dataset/`: Downloaded dataset from Figshare
- `results/`: Directory for saving results (created automatically)

## Prerequisites

- Python 3.x
- Required packages:
  ```
  numpy
  pandas
  matplotlib
  scipy
  scikit-learn
  py-ecg-detectors
  PyWavelets
  ```

You can install all required packages with:
```
pip install numpy pandas matplotlib scipy scikit-learn py-ecg-detectors PyWavelets
```

## Dataset Structure

The dataset from Figshare should be organized as follows:
```
dataset/
  data/
    01/
      BCG/
        01_20231104_BCG.csv
        ...
      Reference/
        RR/
          01_20231104_RR.csv
          ...
    02/
      ...
```

## How to Use

### 1. Download Required Files

1. Download the dataset from [Figshare](https://doi.org/10.6084/m9.figshare.26013157) and extract it to the `dataset/` directory
2. Clone or download the CodeOcean capsule from [here](https://codeocean.com/capsule/1398208/tree) to the `capsule-1398208/` directory

### 2. Process a Test Sample

To test that everything is working properly:
```
python test_sample.py
```

### 3. Process a Single Subject

To process a single subject:
```
python process_single_subject.py [subject_id]
```

Arguments:
- `subject_id`: ID of the subject to process (default: 1)
- You can also modify the script to specify a date or limit processing time

### 4. Process All Subjects

To process all subjects:
```
python process_all_subjects.py [--max_subjects N] [--max_seconds S]
```

Arguments:
- `--max_subjects N`: Maximum number of subjects to process (optional)
- `--max_seconds S`: Maximum number of seconds to process per subject (default: 180)

Example:
```
python process_all_subjects.py --max_subjects 5 --max_seconds 300
```

## Results

The script generates the following results in the `results/` directory:

1. **Individual Subject Results**:
   - `subject_XX_YYYYMMDD_results.csv`: Performance metrics for each subject/date
   - `bcg_processing_subjXX_YYYYMMDD.png`: Visualization of BCG processing steps
   - `j_peaks_close_up_subjXX_YYYYMMDD.png`: Close-up view of detected J-peaks
   - `hr_comparison_subjXX_YYYYMMDD.png`: Comparison of estimated and reference HR
   - `bland_altman_subjXX_YYYYMMDD.png`: Bland-Altman plot for agreement analysis
   - `correlation_subjXX_YYYYMMDD.png`: Correlation plot

2. **Overall Results**:
   - `all_results.csv`: Combined results from all subjects
   - `subject_summary.csv`: Average metrics per subject
   - `summary_report.txt`: Text summary of all results
   - `summary_metrics.png`: Visual summary of performance metrics

## Performance Metrics

The code calculates several metrics to evaluate heart rate estimation:

1. **Mean Absolute Error (MAE)**: Average absolute difference between estimated and reference heart rates
2. **Root Mean Square Error (RMSE)**: Square root of the average of squared differences
3. **Mean Absolute Percentage Error (MAPE)**: Average percentage difference between estimated and reference
4. **Pearson Correlation**: Correlation coefficient between estimated and reference heart rates

## Adapting to Different Datasets

The current implementation is specifically adapted for the ballistocardiogram dataset from Figshare. If you want to use it with a different dataset:

1. Modify the file reading functions in `process_single_subject.py`
2. Adjust the resampling rate in `process_bcg_data()` function to match your data
3. Update the file paths in `process_single_subject.py` and `process_all_subjects.py`

## Limitations

1. The body movement detection from the original capsule is disabled to avoid issues with this specific dataset
2. The current implementation processes only heart rate, not respiratory rate
3. Performance may vary depending on the quality of the BCG signal

## References

1. CodeOcean Capsule (1398208): [https://codeocean.com/capsule/1398208/tree](https://codeocean.com/capsule/1398208/tree)
2. Dataset: Li, Y.-X., Huang, J.-L., Yang, Z.-Y. & Shen, Y.-F. A dataset of ballistocardiogram vital signs with reference sensor signals in natural sleep environments. [https://doi.org/10.6084/m9.figshare.26013157](https://doi.org/10.6084/m9.figshare.26013157) (2024)
3. Sadek, Ibrahim, and Bessam Abdulrazak. "A comparison of three heart rate detection algorithms over ballistocardiogram signals." Biomedical Signal Processing and Control (2021).
4. Sadek, Ibrahim, et al. "A new approach for detecting sleep apnea using a contactless bed sensor: Comparison study." Journal of medical Internet research (2020). 