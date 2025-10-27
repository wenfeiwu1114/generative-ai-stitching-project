[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/QnV1lZz2)

## 1️⃣Data URL

https://www.kaggle.com/datasets/rmisra/news-category-dataset

## 2️⃣Running the Code in Docker
#### A. Build the Docker Image
Navigate to the project directory and build the Docker image:

```bash
docker build -t oqf2740/stitching_image .
```

#### B. Run the Docker Container

```bash
docker run --rm -it oqf2740/stitching_image
```

#### C. Access the Docker Container on the Server
Server: mlds-deepdish2.ads.northwestern.edu

Docker Image Name: oqf2740/stitching_image

#### Run the Docker container:
```bash
docker run --rm -it oqf2740/stitching_image
```

#### 🚀 Running the Unit Test
The unit test ensures the entire system runs correctly and completes within 5 minutes.

Run:

```bash
python3 unit_test_with_lora.py
```

Expected Output
The system retrieves documents.
The generation step produces responses.
Grading agents validate the response.
The test prints the final generated output and stop if the 50 recursion limit is reached.
