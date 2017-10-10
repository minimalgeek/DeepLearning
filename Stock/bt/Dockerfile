FROM python:3.6
ADD ./requirements.txt /src/requirements.txt
RUN pip install -r src/requirements.txt -v

ADD ./notebooks /work
RUN pip uninstall pandas-datareader -y
RUN pip install git+https://github.com/pydata/pandas-datareader.git
WORKDIR /work
CMD jupyter notebook --ip='*' --port=8888 --no-browser --allow-root