FROM python:3.7

ADD . /relevanceai/
WORKDIR /relevanceai/
 
RUN pip install -e .[viz]
