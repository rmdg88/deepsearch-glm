#!/usr/bin/env python
"""Module to load binary files of models and data"""

import json
import os
import subprocess
import requests
from urllib.parse import urljoin


def get_resources_dir():
    """Function to obtain the resources-directory"""

    if "DEEPSEARCH_GLM_RESOURCES_DIR" in os.environ:
        resources_dir = os.getenv("DEEPSEARCH_GLM_RESOURCES_DIR")
    else:
        from deepsearch_glm.andromeda_nlp import nlp_model

        model = nlp_model()
        resources_dir = model.get_resources_path()

    return resources_dir


def list_training_data(key: str, force: bool = False, verbose: bool = False):
    """Function to list the training data"""

    return []


def load_training_data(
    data_type: str, data_name: str, force: bool = False, verbose: bool = False
):
    """Function to load data to train NLP models"""

    assert data_type in ["text", "crf", "fst"]

    resources_dir = get_resources_dir()

    with open(f"{resources_dir}/data.json", "r", encoding="utf-8") as fr:
        training_data = json.load(fr)

    cos_url = training_data["object-store"]
    cos_prfx = training_data["data"]["prefix"]

    done = True
    data = {}

    for name, files in training_data["data"][data_type].items():
        if name == data_name:
            source_url = urljoin(cos_url, f"{cos_prfx}/{files[0]}")
            target_path = os.path.join(resources_dir, files[1])

            if force or not os.path.exists(target_path):
                if verbose:
                    print(f"Downloading {name} from {source_url}...")

                try:
                    response = requests.get(source_url, stream=True)
                    response.raise_for_status()

                    with open(target_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    if verbose:
                        print(f"Downloaded {name} to {target_path}")

                    data[name] = target_path

                except requests.exceptions.RequestException as e:
                    print(f"Failed to download {name}: {e}")
                    done = False
            else:
                if verbose:
                    print(f"Already downloaded {name}")
                data[name] = target_path

    return done, data


def load_pretrained_nlp_models(force: bool = False, verbose: bool = False):
    """Function to load pretrained NLP models"""

    resources_dir = get_resources_dir()

    with open(f"{resources_dir}/models.json", "r", encoding="utf-8") as fr:
        models = json.load(fr)

    cos_url = models["object-store"]
    cos_prfx = models["nlp"]["prefix"]

    downloaded_models = []
    for name, files in models["nlp"]["trained-models"].items():
        source_url = urljoin(cos_url, f"{cos_prfx}/{files[0]}")
        target_path = os.path.join(resources_dir, files[1])

        if force or not os.path.exists(target_path):
            if verbose:
                print(f"Downloading {name} from {source_url}...")

            try:

                response = requests.get(source_url, stream=True)
                response.raise_for_status()

                with open(target_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if verbose:
                    print(f"Downloaded {name} to {target_path}")

                downloaded_models.append(name)

            except requests.exceptions.RequestException as e:
                print(f"Failed to download {name}: {e}")

        elif os.path.exists(target_path):
            if verbose:
                print(f"Already downloaded {name}")
            downloaded_models.append(name)

        else:
            print(f"Missing {name}")

    return downloaded_models
