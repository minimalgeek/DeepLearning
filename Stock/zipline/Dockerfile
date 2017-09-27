FROM python:3.4
ADD ./requirements.txt /src/requirements.txt
RUN pip install -r src/requirements.txt -v
RUN zipline ingest
# it seems it's not necessary, because the newest pandas-datareader fixes everything
#ADD ./other/SPY_benchmark.csv /root/.zipline/data/SPY_benchmark.csv
#ADD ./other/benchmarks.py /usr/local/lib/python3.4/site-packages/zipline/data/benchmarks.py
ADD ./notebooks /work
RUN pip uninstall pandas-datareader -y
RUN pip install git+https://github.com/pydata/pandas-datareader.git
WORKDIR /work
CMD jupyter notebook --ip='*' --port=8888 --no-browser --allow-root