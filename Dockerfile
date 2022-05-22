FROM python:3.7.13
RUN echo "buiding the api image"
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000

RUN mkdir -p /home/projects/save_model/
VOLUME ../save_model/multi_cnn_category_tf1/model_serving:/home/projects/save_model
WORKDIR /home/projects/
RUN echo "working dir is /home/projects/"
COPY ./textMG/startup_api.sh /home/projects/startup_api.sh
CMD ["/home/projects/startup_api.sh"]