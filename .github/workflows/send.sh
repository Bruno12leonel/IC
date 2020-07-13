echo "Criando imagem singularity... ${RECIPE}"
sudo singularity build -F "${RECIPE}.simg" "${RECIPE}"

echo "Configurando ambiente..."
if [[ -z $COLLECTION_CONTAINER ]]; then
  COLLECTION_CONTAINER=collection/container
fi
RCLONE_FILE=~/.config/rclone/rclone.conf
mkdir -p "$(dirname "${RCLONE_FILE}")"

echo "Configurando rclone..."
CLIENT=${CLIENT,,}
case $CLIENT in
google)
  printf "[remote]
type = drive
client_id = %s
client_secret = %s
scope = drive
token = %s
root_folder_id = root
" "$ID_GOOGLE" "$SECRET_GOOGLE" "$TOKEN_GOOGLE" > $RCLONE_FILE
  ;;
aws)
  printf "[remote]
type = s3
provider = AWS
env_auth = false
access_key_id = %s
secret_access_key = %s
region = sa-east-1\n
location_constraint = sa-east-1
" "$ACCESS_KEY_AWS" "$SECRET_KEY_AWS" > $RCLONE_FILE
  ;;
*)
  echo "Plataforma desconhecida: $CLIENT"
  exit
  ;;
esac

echo "Enviando arquivos..."
files=( "${RECIPE}" "${RECIPE}.simg" )
NOW=$(date +'%Y%m%d%H%M%S')
for filename in "${files[@]}"; do
  if [[ -f $filename ]]; then
    path="$(dirname "${filename}")"
    filename="$(basename "${filename}")"
    if [[ "$filename" == *.* ]]; then
      dest="${filename%.*}_${NOW}.${filename##*.}"
    else
      dest="${filename}_${NOW}"
    fi
    rclone copyto "${path}/${filename}" "remote:hpc/containers/${COLLECTION_CONTAINER}/${dest}"
  fi
done