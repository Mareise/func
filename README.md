# Func
This is the fork of the Knative Func repository that is used to run the static-analysis task and to set the needed resource limits to enable GPU execution.

## Requirements
* make
* go 1.24.4
* python 3.10.17
* pyinstaller

## Testing
### Create
```go run main.go create -l python pythontest```
### Deploy
```
cd pythontest
go run ../../main.go deploy --registry index.docker.io/maxireis/ -b=s2i -v --deployment-mode cpu
```

## Deployment
On linux
```
make
chmod +x func
mv func /usr/local/bin
```
