# Using tf-serving on Docker to deploy models

In this particular code I have pulled the tf-serving image from docker and used that to create a container to deploy a simple model.<br>

1) pip install tensorflow-serving-api

2) install docker for desktop from website

3) In command prompt
     
   docker pull tensorflow/serving

4) In command prompt

   docker run -p 8501:8501 --name=tensorflow_serving_container --mount type=bind,source={PATH TO MODEL DIRECTORY},target=/models/{MODEL_NAME} -e MODEL_NAME={MODEL_NAME}  -t tensorflow/serving

5) Make requests using Python requests library 

   url = "http://localhost:8501/v1/models/{MODEL_NAME}:predict"
