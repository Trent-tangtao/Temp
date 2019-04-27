create table s(
  sno bigint not null,
  sname varchar(15),
  birthday date,
  sdepartment varchar(20),
  tel int,
  sex varchar(10),
  primary key (sno)
);
create table c(
  cno bigint not null,
  cname varchar(15),
  teacher varchar(20),
  primary key (cno)
);


# alter table c engine=innodb;
# alter table s engine=innodb;
create table sc(
  sno bigint not null,
  cno bigint not null,
  cscore int not null,
  constraint foreign key (sno) references s(sno) on delete cascade on update cascade ,
# foreign key (cno) references c(cno) on delete set null 此时逻辑关系更好，但是cno不能为not null
  foreign key (cno) references c(cno) on delete  cascade on update cascade
# primary key (sno, cno) 不能做联合主键，否则无法级联删除
);


insert into s(sno,sname,birthday,sdepartment,tel,sex) values (2200120101,'zhangsan','2000-00-01','one',123,'man');
insert into s(sno,sname,birthday,sdepartment,tel,sex) values (2200120102,'one','2000-00-02','two',123,'woman');
insert into s(sno,sname,birthday,sdepartment,tel,sex) values (2200120103,'three','2000-00-03','c',123,'woman');
insert into  c(cno,cname,teacher)  values (001,'c','techer_a');
insert into  c(cno,cname,teacher)  values (002,'second','techer_b');
insert into  c(cno,cname,teacher)  values (003,'third','techer_c');
insert into sc(sno,cno,cscore) values(2200120101,001,90);
insert into sc(sno,cno,cscore) values(2200120102,002,91);
insert into sc(sno,cno,cscore) values(2200120103,003,92);

# (1)
select sname from s where sex like 'wo%';
# (2)
select sno,sname,sdepartment,tel,sex from s where sdepartment like 'c';
# (3)
update sc set cscore = cscore + 5 where sno=2200120102;
# (4)
# 法一，先解除外键
# set foreign_key_checks =0;
# delete from s where sname='zhangsan' ;
# set foreign_key_checks =1;

#法二，级联删除
delete from s where sname='zhangsan' ;