FROM python:3.4
ADD ./requirements.txt /src/requirements.txt
WORKDIR /src
RUN pip install -r requirements.txt
ADD ./notebooks /work
WORKDIR /work
CMD jupyter notebook --ip='*' --port=8888 --no-browser --allow-root