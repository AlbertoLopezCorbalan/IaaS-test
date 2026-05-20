Instalar Docker:
1. sudo apt install -y docker.io
2. sudo systemctl enable docker
3. sudo systemctl start docker
4. sudo usermod -aG docker $USER


Comandos a ejecutar:
1. docker build -t bert-cpu .
2. docker run --rm -v $(pwd)/traces:/app/traces bert-cpu