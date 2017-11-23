FROM base:1.0
ADD . /work
WORKDIR /work
RUN pip install -r requirements.txt
RUN python -m spacy download en
CMD py.test