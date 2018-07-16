#pragma once
#include <map>
#include <string>
class filename
{
public:
	filename();
	virtual ~filename();

	std::string getFuLLName(int fileindex, int qpindex, int qpLevel1, int qplevel2, int warpingscalear, int fillmethodindex);
	//265_t34_Treeflight.mkv
	std::string analyseName(std::string filename);
	std::string findqpfileName(int qpindex, int imageindex, int qplevelindex, int fileindex);
private:
	std::string splitename(std::string filename);
	

	std::map<int, std::string> qp = {
		{264,"264"},
		{265,"265"}
	};

	std::map<int, std::string> qpLevel = {
		{34,"34"},
		{14,"14"}
	};

	std::map<int, std::string> imagetype = {
		{0,"d"},
		{1,"t"}
	};

	std::map<int, std::string> file = {
		{ 1,"balloons"},
		{ 2,"cafe" },
		{ 3,"Dancer" },
		{ 4,"kendo" },
		{ 5,"PoznanCarPark" },
		{ 6,"PoznanHall2" },
		{ 7,"PoznanStreet" },
		{ 8,"ChairBox" },
		{ 9,"GTFly" },
		{ 10,"Shark" },
		{ 11,"Treeflight" },
		{ 12,"family" }
	};

	std::map<int, std::string> method = {
		{ 0,"A1" },
		{ 1,"A2" },
		{ 2,"A3" },
		{ 3,"A4" },
		{ 4,"A5" },
		{ 5,"A6" },
		{ 6,"A7" }
	};

};

