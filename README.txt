Instalar Docker:
1.	sudo dnf install git -y
2.	sudo dnf install -y Docker
3.	sudo systemctl enable Docker
4.	sudo systemctl start Docker
5.	sudo usermod -aG docker $USER


Comandos a ejecutar:
1.	git clone https://github.com/AlbertoLopezCorbalan/IaaS-test
2.	cd ./IaaS-test/
3.	docker build -t bert-cpu .
4.	docker run --rm -v $(pwd)/traces:/app/traces bert-cpu
