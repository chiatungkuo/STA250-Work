create table if not exists data (
	groupid INT,
	value DOUBLE)
row format delimited
fields terminated by '\t'
stored as textfile;

load data local inpath '/home/hadoop/data/groups.txt'
overwrite into table data;

insert overwrite local directory '/home/hadoop/results'
row format delimited
fields terminated by ','
select groupid, avg(value), variance(value) from data group by groupid;

