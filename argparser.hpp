#pragma once

// Standard libraries
#include <exception>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <iomanip>

// Color constants
#define RESET_COLOR "\033[0m"
#define BOLD_COLOR "\033[1m"
#define ITALIC_COLOR "\033[3m"
#define ERROR_COLOR "\033[91;1m"
#define OK_COLOR "\033[92;1m"
#define WARNING_COLOR "\033[93;1m"
#define NOTE_COLOR "\033[94;1m"

// Class for reading command line arguments
class ArgParser {
public:
	// Public aliases
	using Args = std::vector <std::string>;
	using StringMap = std::unordered_map <std::string, std::string>;
	using ArgsMap = std::unordered_map <std::string, int>;
private:
	// Set of all options
	std::set <std::string>	_optns;

	// Does the option take an argument?
	std::set <std::string>	_optn_args;

	// List of all aliases
	std::vector <Args>	_aliases;

	// Map options to aliases
	ArgsMap			_alias_map;

	// Description for each option
	StringMap		_descriptions;

	// Map each option and its argument
	//	empty means that its a boolean
	//	based arg (present/not present)
	StringMap		_matched_args;

	// Positional arguments
	Args			_pargs;

	// Name of the command
	std::string		_name;

	// Number of required positional arguments
	int			_nargs = -1;

	// Helper methods
	bool _optn_arg(const std::string &str) const {
		return _optn_args.find(str) != _optn_args.end();
	}

	bool _optn_present(const std::string &str) const {
		return _optns.find(str) != _optns.end();
	}

	bool _is_optn(const std::string &str) const {
		if (str.empty())
			return false;

		// The second hyphen is a redundant check
		return (str[0] == '-');
	}

	// Set value of option (and aliases)
	void _set_optn(const std::string &str, const std::string &val) {
		int index = _alias_map[str];

		// Set the aliases
		for (const auto &alias : _aliases[index])
			_matched_args[alias] = val;
	}

	// Parse options
	void _parse_option(int argc, char *argv[],
			const std::string &arg, int &i) {
		// Check help first
		if (arg == "-h" || arg == "--help") {
			help();
			exit(0);
		}

		// Check if option is not present
		if (!_optn_present(arg)) {
			fprintf(stderr, "%s%s: %serror:%s unknown option %s\n",
				BOLD_COLOR, _name.c_str(), ERROR_COLOR,
				RESET_COLOR, arg.c_str());
			exit(-1);
		}

		// Handle arguments
		if (_optn_arg(arg)) {
			if ((++i) >= argc) {
				fprintf(stderr, "%s%s: %serror:%s option %s need an argument\n",
					BOLD_COLOR, _name.c_str(),
					ERROR_COLOR, RESET_COLOR,
					arg.c_str());
				exit(-1);
			}

			_set_optn(arg, argv[i]);
		} else {
			_set_optn(arg, "");
		}
	}

	// Convert string to value for methods
	template <class T>
	T _convert(const std::string &) const {
		return T();
	}
public:
	// Option struct

	// TODO: option to provide argument name in help
	struct Option {
		Args		aliases;
		std::string	descr;
		bool		arg;
		// TODO: bool optional, etc.
		// TODO: defaults...

		// Constructor
		Option(const std::string &str, const std::string &descr = "",
			bool arg = false) : aliases {str}, descr(descr), arg(arg) {}

		Option(const Args &aliases, const std::string &descr = "",
			bool arg = false) : aliases(aliases), descr(descr), arg(arg) {}
	};

	// Number of required positional arguments
	ArgParser(const std::string &name = "", int nargs = -1)
			: _name(name), _nargs(nargs) {
		add_optn(Args {"-h", "--help"}, "show this message");
	}

	// Full constructor, with all options
	ArgParser(const std::string &name, int nargs,
			const std::vector <Option> &opts) : ArgParser(name, nargs) {
		for (const auto &opt : opts)
			add_optn(opt.aliases, opt.descr, opt.arg);
	}

	// Add an option
	void add_optn(const std::string &str, const std::string &descr = "",
			bool arg = false) {
		// Add the option
		_optns.insert(str);

		// Add option to those which take an argument
		if (arg)
			_optn_args.insert(str);

		// Add the description
		_descriptions[str] = descr;

		// Add the alias
		_aliases.push_back(Args {str});

		// Alias map to nullptr
		_alias_map[str] = (int) _aliases.size() - 1;
	}

	// Add an option with aliases
	void add_optn(const Args &args, const std::string &descr = "",
			bool arg = false) {
		// Add the options
		for (const auto &arg : args)
			_optns.insert(arg);

		// Add option to those which take an argument
		if (arg) {
			for (const auto &arg : args)
				_optn_args.insert(arg);
		}

		// Add the descriptions
		for (const auto &arg : args)
			_descriptions[arg] = descr;

		// Add the aliases
		_aliases.push_back(args);

		// Add to the alias map
		for (const auto &arg : args)
			_alias_map[arg] = (int) _aliases.size() - 1;
	}

	void parse(int argc, char *argv[]) {
		// Set name if empty
		if (_name.empty())
			_name = argv[0];

		// Process the arguments
		for (int i = 1; i < argc; i++) {
			std::string arg = argv[i];
			if (_is_optn(arg))
				_parse_option(argc, argv, arg, i);
			else
				_pargs.push_back(arg);
		}

		// Check number of positional args
		if (_nargs > 0 && _pargs.size() < _nargs) {
			fprintf(stderr, "%s%s: %serror:%s requires %d argument%c,"
				" was only provided %lu\n",
				BOLD_COLOR, _name.c_str(),
				ERROR_COLOR, RESET_COLOR,
				_nargs, 's' * (_nargs != 1),
				_pargs.size());
			exit(-1);
		}
	}

	const Args &pargs() const {
		return _pargs;
	}

	// Retrieving positional arguments
	template <class T = std::string>
	inline T get(size_t i) const {
		return _convert <T> (_pargs[i]);
	}

	// Retrieve optional arguments
	template <class T = std::string>
	inline T get_optn(const std::string &str) {
		// Check if its a valid option
		if (_optns.find(str) == _optns.end())
			throw bad_option(str);

		// Check if the option even takes arguments
		if (!_optn_arg(str))
			throw optn_no_args(str);

		// Check if the option value is null
		if (_matched_args.find(str) == _matched_args.end())
			throw optn_null_value(str);

		// Return the converted value
		return _convert <T> (_matched_args[str]);
	}

	// Print error as a command
	int error(const std::string &str) const {
		fprintf(stderr, "%s%s: %serror:%s %s\n",
			BOLD_COLOR, _name.c_str(), ERROR_COLOR,
			RESET_COLOR, str.c_str());
		return -1;
	}

	// Print warning as a command
	int warning(const std::string &str) const {
		fprintf(stderr, "%s%s: %swarning:%s %s\n",
			BOLD_COLOR, _name.c_str(), WARNING_COLOR,
			RESET_COLOR, str.c_str());
		return 0;
	}

	// Print help
	void help() {
		// Print format of command
		printf("usage: %s", _name.c_str());
		for (const Args &aliases : _aliases) {
			// Just use the first alias
			std::string optn = aliases[0];
			printf(" [%s%s]", optn.c_str(),
				_optn_arg(optn) ? " arg" : "");
		}
		printf("\n");

		// Stop if no optional arguments
		if (_optns.empty())
			return;

		// Print description
		printf("\noptional arguments:\n");
		for (const Args &alias : _aliases) {
			std::string combined;
			for (size_t i = 0; i < alias.size(); i++) {
				combined += alias[i];

				if (i != alias.size() - 1)
					combined += ", ";
			}

			printf("  %*s", 20, combined.c_str());

			std::string descr = _descriptions[alias[0]];
			if (descr.empty())
				printf(" [?]\n");
			else
				printf(" %s\n", descr.c_str());
		}
	}

	// For debugging
	void dump() {
		std::cout << "Positional arguments: ";
		for (size_t i = 0; i < _pargs.size(); i++) {
			std::cout << "\"" << _pargs[i] << "\"";
			if (i + 1 < _pargs.size())
				std::cout << ", ";
		}
		std::cout << std::endl;

		for (const Args &alias : _aliases) {
			std::string combined;
			for (size_t i = 0; i < alias.size(); i++) {
				combined += alias[i];

				if (i != alias.size() - 1)
					combined += ", ";
			}

			std::cout << "\t" << std::left << std::setw(20)
				<< combined << " ";

			std::string optn = alias[0];
			if (_matched_args.find(optn) == _matched_args.end()) {
				std::cout << "Null\n";
			} else {
				std::string value = _matched_args[optn];

				std::cout << (value.empty() ? "Present" : value) << "\n";
			}
		}
	}

	// Thrown if not an option
	class bad_option : public std::runtime_error {
	public:
		bad_option(const std::string &str) :
			std::runtime_error("ArgParser: has no registered"
				" option \"" + str + "\"") {}
	};

	// Thrown if the option does not take an argument
	class optn_no_args : public std::runtime_error {
	public:
		optn_no_args(const std::string &str) :
			std::runtime_error("ArgParser: option \"" +
				str + "\" does not take arguments") {}
	};

	// Thrown if the option is null
	class optn_null_value : public std::runtime_error {
	public:
		optn_null_value(const std::string &str) :
			std::runtime_error("ArgParser: option \"" +
				str + "\" has null value (not specified)") {}
	};
};

template <>
inline std::string ArgParser::_convert <std::string> (const std::string &str) const {
	return str;
}

// Integral types
template <>
inline long long int ArgParser::_convert <long long int> (const std::string &str) const {
	return std::stoll(str);
}

template <>
inline long int ArgParser::_convert <long int> (const std::string &str) const {
	return _convert <long long int> (str);
}

// TODO: integral specializations...
template <>
inline int ArgParser::_convert <int> (const std::string &str) const {
	return _convert <long long int> (str);
}

// Floating types
template <>
inline long double ArgParser::_convert <long double> (const std::string &str) const {
	return std::stold(str);
}

template <>
inline double ArgParser::_convert <double> (const std::string &str) const {
	return _convert <long double> (str);
}

template <>
inline float ArgParser::_convert <float> (const std::string &str) const {
	return _convert <long double> (str);
}

// Boolean conversion
template <>
inline bool ArgParser::_convert <bool> (const std::string &str) const {
	return str == "true" || str == "1";
}

// Get option for booleans
//	special case because options that take no
//	arguments are true or false depending on
//	whether the option is present or not
template <>
inline bool ArgParser::get_optn <bool> (const std::string &str) {
	// Check if its a valid option
	if (_optns.find(str) == _optns.end())
		throw bad_option(str);

	// If its empty and no argument, return true
	if (!_optn_arg(str)) {
		if (_matched_args.find(str) == _matched_args.end())
			return false;
		if (_matched_args[str].empty())
			return true;
		throw optn_no_args(str);
	}

	// Check if the option value is null
	if (_matched_args.find(str) == _matched_args.end())
		throw optn_null_value(str);

	// Return the converted value
	return _convert <bool> (_matched_args[str]);
}
