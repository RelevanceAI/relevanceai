FROM python:3.8

ADD . /relevanceai/
WORKDIR /relevanceai/
 
RUN pip install -e .[viz]
