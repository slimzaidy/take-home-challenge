FROM frolvlad/alpine-miniconda3:python3.7

COPY requirements_prod.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements_prod.txt && \
 	rm requirements_prod.txt

EXPOSE 8000

COPY . .

CMD ["uvicorn", "app.model_app.server:app", "--host", "0.0.0.0", "--port", "8000"]