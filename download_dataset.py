import os
import multiprocessing

def download_parquet_file(file_number, token):
    url = f"https://huggingface.co/datasets/CarperAI/pilev2-dev/resolve/main/data/pubmed/PubMed_{file_number}.parquet"
    command = f'wget --header="Authorization: Bearer {token}" "{url}"'
    os.system(command)
    print(f'Downloaded PubMed_{file_number}.parquet')


if __name__ == "__main__":
    HF_TOKEN = "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"  # Replace with your Hugging Face token
    start_file = 0
    end_file = 46
    
    # Create a pool of processes to download files in parallel
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    # Use the pool to download the files in parallel
    for file_number in range(start_file, end_file + 1):
        pool.apply_async(download_parquet_file, args=(file_number, HF_TOKEN))
    
    pool.close()
    pool.join()

    print("All downloads completed.")
