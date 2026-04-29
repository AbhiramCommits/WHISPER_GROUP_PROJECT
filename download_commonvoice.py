from datacollective import download_dataset
import datacollective

dataset_id = 'cmndapwry02jnmh07dyo46mot'
# print(get_dataset_details(dataset_id))
dataset_path = download_dataset(dataset_id, download_directory='cv-data', show_progress=True, overwrite_existing=False)
print(dataset_path)