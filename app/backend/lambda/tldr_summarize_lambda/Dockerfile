FROM public.ecr.aws/lambda/python:3.8.2023.07.03.13
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY lambda_function.py .
ADD model model
CMD [ "/app/lambda_function.lambda_handler" ]


