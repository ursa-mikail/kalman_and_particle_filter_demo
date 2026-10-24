FROM golang:1.22-alpine AS b
WORKDIR /src
COPY . .
RUN go mod tidy && CGO_ENABLED=0 go build -o /api .
FROM alpine:3.20
COPY --from=b /api /api
EXPOSE 8080
CMD ["/api"]
