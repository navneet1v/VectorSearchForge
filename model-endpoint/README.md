https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container


## Local Testing
```
docker build  -t model-endpoint-test:latest .
```

```
docker run -p 8080:8080 model-endpoint-test:latest
```