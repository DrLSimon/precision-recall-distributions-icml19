# Revisiting Precision Recall Definition for Generative Modeling

Official code for [Revisiting precision recall definition for generative modeling](http://proceedings.mlr.press/v97/simon19a/simon19a.pdf) by [Loic Simmon](https://simonl02.users.greyc.fr/), [Ryan Webster](https://github.com/ryanwebster90), and [Julien Rabin](https://sites.google.com/site/rabinjulien/), presented at [ICML 2019](https://icml.cc/Conferences/2019). The poster can be downloaded [here](https://s3.amazonaws.com/postersession.ai/62b450e7-401d-41bf-be0a-93cf06c0885b.pdf) and the highlight video can be watched [here](https://youtube.videoken.com/embed/HlyE7P7gxYE?tocitem=35).

_Note that this code was adapted from [this github project](https://github.com/msmsajjadi/precision-recall-distributions) corresponding to [Assessing Generative Models via Precision and Recall](https://arxiv.org/abs/1806.00035). Besides the new method, the original method was also reimplemented in pytorch (in order to remove tensorflow dependency). Doing so we obtained slightly different curves compared to the orginal tensorflow implementation._

## Usage

### Requirements
A list of required packages is provided in [requirements_minimal.txt](requirements_minimal.txt) and may be installed by running:
```shell
pip install -r requirements_minimal.txt
```

Alternatively, you may find a yaml conda environment file [conda_env.yaml](conda_env.yaml) that can be used as follows:
```shell
# If you want to complete your current environment
conda env update -f conda_env.yaml
# Or if you want a fresh new environment (named here prdenv)
conda env create -n prdenv -f conda_env.yaml
conda activate prdenv
```

### Automatic: Compute PRD for folders of images on disk
Example: you have a folder of images from your true distribution (e.g., `~/real_images/`) and any number of folders of generated images (e.g., `~/generated_images_1/` and `~/generated_images_2/`). Note that the number of images in each folder needs to be the same.

In a shell, cd to the repository directory and run
```shell
python prd_from_image_folders.py --classif --reference_dir ~/real_images/ --eval_dirs ~/generated_images_1/ ~/generated_images_2/ --eval_labels model_1 model_2 # ICML'19 paper version
python prd_from_image_folders.py --reference_dir ~/real_images/ --eval_dirs ~/generated_images_1/ ~/generated_images_2/ --eval_labels model_1 model_2           # Original NeurIPS'18 paper version
```

Besides a [dataset folder](datasets) was provided along with a script [runCifarModes.sh](runCifarModes.sh) to reproduce the Cifar modes experiment from the paper.

For further customization, run `./prd_from_image_folders.py -h` to see the list of available options.


## BibTex citation
```
@InProceedings{pmlr-v97-simon19a,
  title = 	 {Revisiting precision recall definition for generative modeling},
  author = 	 {Simon, Loic and Webster, Ryan and Rabin, Julien},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  year = 	 {2019}
  }
```

## Further information
External copyright for: [prd_score.py](https://github.com/msmsajjadi/precision-recall-distributions/blob/master/prd_score.py) and [prd_from_image_folders.py](https://github.com/msmsajjadi/precision-recall-distributions/blob/master/prd_from_image_folders.py)  goes to [Mehdi S. M. Sajjadi](http://msajjadi.com)
Copyright for remaining files: [Loic Simon](https://simonl02.users.greyc.fr/)<br>

License for all files: [Apache License 2.0](LICENSE)

For any questions, comments or help to get it to run, please don't hesitate to mail us: <loic.simon@ensicaen.fr>
