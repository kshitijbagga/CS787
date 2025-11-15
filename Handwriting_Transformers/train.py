import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import time
import torch
import numpy as np
import pickle  # <-- Added for saving pickle files
from itertools import cycle
from tqdm import tqdm
from scipy import linalg

from data.dataset import TextDataset, TextDatasetval
from models import create_model
from models.model import TRGAN
from params import *
from torch import nn


def main():
    # --- Initialize project ---
    init_project()

    # --- Load Training Dataset ---
    TextDatasetObj = TextDataset(num_examples=NUM_EXAMPLES)
    dataset = torch.utils.data.DataLoader(
        TextDatasetObj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=TextDatasetObj.collate_fn,
    )

    # --- Load Validation Dataset ---
    TextDatasetObjval = TextDatasetval(num_examples=NUM_EXAMPLES)
    datasetval = torch.utils.data.DataLoader(
        TextDatasetObjval,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=TextDatasetObjval.collate_fn,
    )

    # --- Initialize Model ---
    model = TRGAN()

    os.makedirs("saved_models", exist_ok=True)
    MODEL_PATH = os.path.join("saved_models", EXP_NAME)

    if os.path.isdir(MODEL_PATH) and RESUME:
        model.load_state_dict(torch.load(MODEL_PATH + "/model.pth"))
        print(MODEL_PATH + " : Model loaded successfully")
    else:
        if not os.path.isdir(MODEL_PATH):
            os.mkdir(MODEL_PATH)

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        start_time = time.time()

        progress_bar = tqdm(
           enumerate(dataset),
           total=len(dataset),
           desc=f"Epoch {epoch+1}/{EPOCHS}",
        )

        for i, data in enumerate(dataset):
            # --- Training steps ---
            if (i % NUM_CRITIC_GOCR_TRAIN) == 0:
                model._set_input(data)
                model.optimize_G_only()
                model.optimize_G_step()

            if (i % NUM_CRITIC_DOCR_TRAIN) == 0:
                model._set_input(data)
                model.optimize_D_OCR()
                model.optimize_D_OCR_step()

            if (i % NUM_CRITIC_GWL_TRAIN) == 0:
                model._set_input(data)
                model.optimize_G_WL()
                model.optimize_G_step()

            if (i % NUM_CRITIC_DWL_TRAIN) == 0:
                model._set_input(data)
                model.optimize_D_WL()
                model.optimize_D_WL_step()

            # --- Update progress bar losses ---
            losses = model.get_current_losses()
            progress_bar.set_postfix(
                {
                   "G": f"{losses['G']:.4f}",
                   "D": f"{losses['D']:.4f}",
                   "OCR_f": f"{losses['OCR_fake']:.4f}",
                   "OCR_r": f"{losses['OCR_real']:.4f}",
                   "W_fake": f"{losses['w_fake']:.4f}",
               }
           )

        end_time = time.time()

        # --- Validation step ---
        data_val = next(iter(datasetval))
        page = model._generate_page(model.sdata, model.input["swids"])
        page_val = model._generate_page(
            data_val["simg"].to(DEVICE), data_val["swids"]
        )

        print(
            f"\n[Epoch {epoch+1}] Time: {end_time - start_time:.2f}s"
        )

        # --- Save Model ---
        if epoch % SAVE_MODEL == 0:
            torch.save(model.state_dict(), MODEL_PATH + "/model.pth")

            # 💾 Save full model as pickle
            with open(MODEL_PATH + "/model.pkl", "wb") as f:
                pickle.dump(model, f)

        # if epoch % SAVE_MODEL_HISTORY == 0:
        #     torch.save(
        #         model.state_dict(), MODEL_PATH + f"/model{epoch}.pth"
        #     )

        #     # 💾 Save full model snapshot
        #     with open(MODEL_PATH + f"/model{epoch}.pkl", "wb") as f:
        #         pickle.dump(model, f)


if __name__ == "__main__":
    main()
