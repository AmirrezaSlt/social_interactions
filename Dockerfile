FROM golang:1.11
WORKDIR /go/src/app
COPY . .
RUN go get -u github.com/gorilla/mux &&\
    go get go.mongodb.org/mongo-driver/mongo
    
