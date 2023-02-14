import numpy as np
import torch
from torch.utils.data import DataLoader

from model import FontClassifierModel
from synth_text_dataset import SynthTextCharactersDatasetTest
from transforms import char_transform, img_transform

my_labels = {
    "Titillium Web": 0,
    "Alex Brush": 1,
    "Ubuntu Mono": 2,
    "Open Sans": 3,
    "Sansation": 4,
}
csv_labels = {
    "Open Sans": 0,
    "Sansation": 1,
    "Titillium Web": 2,
    "Ubuntu Mono": 3,
    "Alex Brush": 4,
}
labels_transformation = {
    my_labels["Titillium Web"]: csv_labels["Titillium Web"],
    my_labels["Alex Brush"]: csv_labels["Alex Brush"],
    my_labels["Ubuntu Mono"]: csv_labels["Ubuntu Mono"],
    my_labels["Open Sans"]: csv_labels["Open Sans"],
    my_labels["Sansation"]: csv_labels["Sansation"],
}


def my_label_to_csv_label(label):
    return labels_transformation[label]


def main():
    filename = r"Project - Test Set\SynthText_test.h5"
    model_name = (
        r"all_models_with_perms\FontClassifierModel_0.01_50_CrossEntropyLoss_SGD_32_0.8"
    )

    init_shape = (100, 100)
    test_dataset = SynthTextCharactersDatasetTest(
        filename=filename,
        shape=init_shape,
        char_transform=char_transform,
        full_image_transform=img_transform,
    )

    model = FontClassifierModel(init_shape=init_shape, in_channels=1)
    model.load_state_dict(torch.load(rf"models\{model_name}.pth"))
    model.eval()

    test_dataloader = DataLoader(test_dataset)
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    model.to(device)

    char_pred = np.zeros(len(test_dataset.chars))
    with torch.no_grad():
        for idx, X in enumerate(test_dataloader):
            X = X.to(device)[0]
            curr_y_pred = model(X).argmax(dim=1)
            curr_char_pred = np.bincount(curr_y_pred.cpu()).argmax()
            for i in test_dataset.get_word_indexes_at_idx(idx):
                char_pred[i] = curr_char_pred

    with open("test_labels.csv", "w") as csv_results:
        csv_results.write(
            " ,image,char,Open Sans,Sansation,Titillium Web,Ubuntu Mono,Alex Brush\n"
        )
        for i in range(len(test_dataset.chars)):
            img_name = test_dataset.get_img_name_at_idx(i)
            char = test_dataset.get_char_at_idx(i)
            char = char if char != "," else '","'
            char = char if char != '"' else '""""'
            csv_label = my_label_to_csv_label(char_pred[i])
            csv_results.write(
                f"{i},{img_name},{char},{','.join(['0' if i != csv_label else '1' for i in range(5)])}\n"
            )


if __name__ == "__main__":
    main()
