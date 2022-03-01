# Partea Client CLI

Instead of using the frontend client, a CLI can also be used to perform the federated and privacy-aware time-to-event analysis.

For now, it is available as a docker image.

## 1. Install Docker

Follow the installation instructions for your OS: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

## 2. Pull the image
`docker pull registry.blitzhub.io/partea-cli`

## 3. Run 
`docker run -d -v PATH/TO/YOUR/FILE:/mnt/input registry.blitzhub.io/partea-cli -u USERNAME -pw PASSWORD -token TOKEN -path /mnt/input/FILENAME -duration_col DURATION_COLUMN -event_col EVENT_COLUMN -sep ,`

## Help
To See all available options run `docker run -d -v PATH/TO/YOUR/FILE:/mnt/input registry.blitzhub.io/partea-cli`

```
usage: main.py [-h] -u USERNAME -token TOKEN -path PATH [-pw [PASSWORD]] [-server_url [SERVER_URL]] -duration_col [DURATION_COL] -event_col
               [EVENT_COL] [-category_col [CATEGORY_COL]] [-sep [SEP]]
```