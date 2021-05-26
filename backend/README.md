## Metron Backend

### Features:
* Image Forgery Detection

### Installation

1. Install Docker
1. Build the Docker image:
```docker image build --tag metron_backend .```
1. Run the App:
```docker container run --publish 5000:5000 metron_backend```

Note: Stop the app using Ctrl-C.

## Usage

```
Path: /v1/fake_checker
Method: POST
Body: JSON
{
    "image_url": "http://...."
}
Result:
{
    "fake": <boolean>
}
```
