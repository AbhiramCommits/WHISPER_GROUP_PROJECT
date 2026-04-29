from datacollective import download_dataset
from dotenv import load_dotenv

load_dotenv()


dataset_id = 'cmndapwry02jnmh07dyo46mot'
# print(get_dataset_details(dataset_id))
dataset_path = download_dataset(dataset_id, show_progress=True, overwrite_existing=False)
print(dataset_path)
