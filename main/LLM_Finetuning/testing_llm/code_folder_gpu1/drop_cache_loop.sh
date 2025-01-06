while true; do
free -h
echo "cache dropped"
sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches "
free -h
sleep 300
#wait for 5 mins
done
