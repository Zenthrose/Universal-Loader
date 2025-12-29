FROM ubuntu:22.04

RUN apt-get update && apt-get install -y cmake ninja-build libvulkan-dev python3 python3-pip

COPY . /app
WORKDIR /app

RUN pip install pybind11

RUN mkdir build && cd build && cmake .. -G Ninja && ninja

CMD ["./build/benchmark", "--help"]