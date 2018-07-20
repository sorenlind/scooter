FROM python:3.6-slim

WORKDIR /scooter
ADD . /scooter

RUN pip3 install .
RUN pip3 install gunicorn

EXPOSE 80

CMD ["gunicorn", "-b", "0.0.0.0:80", "wsgi:app"]