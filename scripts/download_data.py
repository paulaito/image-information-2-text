from image_information_retrieval.utils.downloader import download_hf_sample_data

download_hf_sample_data("HuggingFaceM4/the_cauldron", "diagram_image_to_text", split="train", num_samples=15)
download_hf_sample_data("HuggingFaceM4/the_cauldron", "chart2text", split="train", num_samples=15)

#download_hf_sample_data("HuggingFaceM4/the_cauldron", "ai2d", split="train", num_samples=15)