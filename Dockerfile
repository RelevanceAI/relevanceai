FROM python:3.8

ADD . /relevanceai/
WORKDIR /relevanceai/
 
RUN python setup.py install

CMD ["python"]