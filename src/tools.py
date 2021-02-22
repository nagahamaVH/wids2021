import pandas as pd


def make_submission(predictions, test_path, file_name):
    '''Creates a .csv submission file.'''

    # Create submission
    sample_submission = pd.read_csv(test_path)
    IDs = sample_submission["encounter_id"]

    to_submit = {'encounter_id': IDs, 'diabetes_mellitus': predictions}
    df_to_submit = pd.DataFrame(to_submit).set_index(['encounter_id'])

    df_to_submit.to_csv(file_name)
    print("Submission ready.")
