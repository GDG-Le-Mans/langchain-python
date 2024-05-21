FROM python:3.9-slim
WORKDIR /app/
COPY . .
ENV PYHTONUNBUFFERED=1
RUN pip3 install langchain_community python-dotenv pypdf chromadb
CMD ["python3","-m","demo.gdg"]