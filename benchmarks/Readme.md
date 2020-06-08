# Add benchmarks 
The current version includes the following datasets/benchmarks:
* `OpenLORIS-Object`
* `Permuated MNIST`
* `CIFAR-100`

## Data Preparation


### OpenLORIS-Object
Step 1: Download data (including RGB-D images, masks, and bounding boxes) following [this instruction](https://drive.google.com/open?id=1KlgjTIsMD5QRjmJhLxK4tSHIr0wo9U6XI5PuF8JDJCo). 

Step 2: Run following scripts:
```
 python3 benchmark1.py
 python3 benchmark2.py
```

Step 3: Put train/test/validation file under `./data/OpenLORIS-Object`. For more details, please follow `note` file under each sub-directories in `./data/OpenLORIS-Object`.

Step 4: Generate the `.pkl` files of data.
```
 python3 pk_gene.py
 python3 pk_gene_sequence.py
```

### Permuated MNIST
Run following scripts.
'''
cd MNIST
sh mnist.sh
'''

### CIFAR-100
Run following scripts.
'''
cd CIFAR
sh cifar.sh
'''

Benchmark results:
