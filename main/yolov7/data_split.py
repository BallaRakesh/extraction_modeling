import glob

if __name__ == "__main__":
    # performa invoices
    # image_folder: str = "/New_Volume/trade-finance/final_delivery/pi/data/V2/performa_invoice"

    # purchase order
    image_folder: str = "/home/tarun/Downloads/FooterData/combined"

    # bg cancellation
    # image_folder: str = "/New_Volume/trade-finance/final_delivery/bg_cancellation/data/V2"

    labelled_files = [
        f"{image_names.split('/')[-1]}\n"
        for image_names in glob.glob(f"{image_folder}/Labels/*.txt")
    ]

    images = [f"{image_names.split('/')[-1]}\n"
        for image_names in glob.glob(f"{image_folder}/Images/*.png")
    ]

    common_labels = {
        labels.split(".txt")[0] for labels in labelled_files
    }.intersection({image.split(".png")[0] for image in images})

    
    common_labels  = [f"{label}.txt\n" for label in common_labels]
    
    # printing some of the labels for testing
    print(common_labels[:3])
        
    # common labels lenght
    print(f"len of common labels: {len(common_labels)}")
    # exit("+++++++++")
    
    ratio_val :float = 0.2

    train_samples_imgs = common_labels[:-int(ratio_val * len(common_labels))]
    print(len(train_samples_imgs))

    with open(f"{image_folder}/train.txt", "w") as file:
        file.writelines(train_samples_imgs)


    test_samples_imgs = common_labels[-int(ratio_val * len(common_labels)):]
    print(len(test_samples_imgs))


    with open(f"{image_folder}/test.txt", "w") as file:
        file.writelines(test_samples_imgs)
    

