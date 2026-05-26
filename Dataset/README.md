# Dataset Placement

The original medical imaging dataset is not published in this repository.

The bundled trained weights are sufficient for uploading an image and running
the three-model inference application. To reproduce evaluation or training,
place an authorized copy of the dataset here:

```text
Dataset/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

Do not publish patient data or restricted clinical images without appropriate
authorization and de-identification.
