
HOSTNAME = '172.16.33.237';
USERNAME = 'cidi';
PASSWORD = '123';

ssh2_conn = ssh2_config(HOSTNAME,USERNAME,PASSWORD);

ssh2_conn = ssh2_command(ssh2_conn, 'ls /; ls ~',1);
ssh2_conn = ssh2_close(ssh2_conn);
